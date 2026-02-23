"""
agents/got_nodes.py — Graph of Thoughts reasoning layer.

THREE NODES, ONE LOOP:

  generate_hypotheses_node   — STINKMEANER and SAMUEL read the specialist
                               data and each produce a complete, independent
                               trading hypothesis as a ThoughtNode.

  adversarial_critique_node  — GRANDDAD attacks each hypothesis with a series
                               of stress tests (flash crash, volume spoof, macro
                               shock, liquidity drain). Thoughts whose confidence
                               drops below the survival threshold are pruned.
                               All attacks are recorded for the audit trail.

  aggregate_and_improve_node — CLAYTON and JULIUS read the surviving thoughts,
                               mathematically merge them into a consensus, and
                               write a richer TradeRecommendation back to state.
                               The recommendation now includes which attacks it
                               survived — invaluable context for the human
                               reviewer in the approval dashboard.

LOOP BEHAVIOUR:
  If the adversary prunes all hypotheses (nothing survives), the graph
  routes back to generate for another pass. got_generation increments.
  At got_max_generations the system writes a HOLD recommendation with an
  honest rationale: "All hypotheses failed stress testing." This hard
  ceiling prevents infinite loops when market conditions are genuinely
  ambiguous.

DESIGN PRINCIPLE — SEPARATION OF CONCERNS:
  These nodes do NOT call external APIs. They reason over data that the
  four specialist agents already fetched and stored in state. This keeps
  the GoT layer fast (no network latency) and focused: its job is to
  think harder about existing data, not to gather more of it.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from state import AdversarialAttack, GoTState, ThoughtNode

# ── Survival threshold ───────────────────────────────────────────────────────
# A thought whose confidence_score drops at or below this value after the
# adversarial critique is pruned from the graph. 0.55 is deliberately low —
# we want to keep thoughts that are "probably right under stress" rather than
# only keeping near-certain ones. In highly volatile markets, 0.55 may
# correctly be the ceiling for any directional thesis.
SURVIVAL_THRESHOLD = 0.55

# ── Model factory ────────────────────────────────────────────────────────────
def _llm(model: str = "claude-sonnet-4-6") -> ChatAnthropic:
    return ChatAnthropic(model=model, temperature=0.2, max_tokens=4096)


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Generate Hypotheses
# Personas: STINKMEANER (aggressive momentum) and SAMUEL (disciplined trend)
# ─────────────────────────────────────────────────────────────────────────────

STINKMEANER_SYSTEM = """\
You are STINKMEANER — the aggressive momentum hunter in this trading system.
You thrive on high volatility and hunt breakout opportunities others miss.
Your cognitive style is bold, conviction-driven, and offensive.

You have access to the specialist agents' analysis. Your job is to generate
ONE complete trading hypothesis in JSON, capturing the most compelling
bullish or bearish momentum case for this asset right now.

Be specific: cite actual RSI values, price levels, and volume data from the
analysis. Do not be vague. A hypothesis with no specific numbers is not useful.

Output ONLY a JSON object inside ```json ... ``` with this exact structure:
{
  "thesis": "one-sentence directional call",
  "direction": "long" | "short" | "neutral",
  "timeframe": "1d" | "3d" | "1w" | "2w",
  "entry_rationale": "specific technical/momentum reason to enter now",
  "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"],
  "proposed_entry": float,
  "proposed_target": float,
  "proposed_stop": float,
  "primary_risk": "the single biggest thing that would invalidate this thesis",
  "confidence_score": float  // your honest 0.0-1.0 conviction level
}
"""

SAMUEL_SYSTEM = """\
You are SAMUEL — the cold, disciplined trend analyst in this trading system.
You demand chart confirmation and mathematical backing before any call.
Your cognitive style is precise, systematic, and patient.

You have access to the specialist agents' analysis. Your job is to generate
ONE complete trading hypothesis in JSON, capturing the most defensible
trend-based case for this asset right now.

Be specific: RSI values, MACD crossover status, EMA positions, support/
resistance levels. If the technical picture is mixed, say so clearly — a
HOLD thesis is a valid output and better than a forced directional call.

Output ONLY a JSON object inside ```json ... ``` with this exact structure:
{
  "thesis": "one-sentence directional call",
  "direction": "long" | "short" | "neutral",
  "timeframe": "1d" | "3d" | "1w" | "2w",
  "entry_rationale": "specific technical reason to enter now",
  "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"],
  "proposed_entry": float,
  "proposed_target": float,
  "proposed_stop": float,
  "primary_risk": "the single biggest thing that would invalidate this thesis",
  "confidence_score": float  // your honest 0.0-1.0 conviction level
}
"""


def _build_data_summary(state: GoTState) -> str:
    """
    Summarise the specialist agents' outputs into a concise context block
    that STINKMEANER and SAMUEL can reason over.
    """
    md = state.get("market_data", {})
    ta = state.get("technical_analysis", {})
    ra = state.get("risk_assessment", {})
    gen = state.get("got_generation", 1)

    prior_note = ""
    if gen > 1:
        # Tell the generators why we're on a retry loop
        pruned = [
            t for t in state.get("thought_graph", [])
            if not t.get("survived_critique", False)
        ]
        if pruned:
            reasons = [
                note
                for t in pruned
                for note in t.get("critique_notes", [])
            ][:4]
            prior_note = (
                f"\n\n⚠️  RETRY PASS {gen}: Previous hypotheses were pruned by "
                f"the adversary. Generate a fundamentally different thesis. "
                f"Pruning reasons from last pass:\n"
                + "\n".join(f"  - {r}" for r in reasons)
            )

    return f"""SPECIALIST ANALYSIS SUMMARY{prior_note}

MARKET DATA:
  Symbol:           {md.get('symbol', state.get('target_symbol', '?'))}
  Price:            ${md.get('price_usd', 0):,.4f}
  24h Change:       {md.get('price_change_24h', 0):+.2f}%
  Volume 24h:       ${md.get('volume_24h', 0):,.0f}
  Market Cap:       ${md.get('market_cap', 0):,.0f}
  Sentiment:        {md.get('sentiment', 'unknown')}
  Fear & Greed:     {md.get('fear_greed_index', '?')} / 100
  Headlines:        {'; '.join((md.get('news_headlines') or [])[:3])}

TECHNICAL PICTURE:
  RSI (14):         {ta.get('rsi', '?')}
  MACD Signal:      {ta.get('macd_signal', '?')}
  EMA Trend:        {ta.get('ema_trend', '?')}
  BB Position:      {ta.get('bb_position', '?')}
  Support:          {ta.get('support_levels', [])}
  Resistance:       {ta.get('resistance_levels', [])}
  Summary:          {ta.get('summary', 'No summary available.')}

RISK PARAMETERS:
  Risk Level:       {ra.get('risk_level', '?')}
  Max Position:     ${ra.get('max_position_size_usd', 0):,.0f} ({ra.get('max_position_size_pct', 0):.1f}%)
  Stop-Loss:        -{ra.get('stop_loss_pct', 0):.1f}%
  Take-Profit:      +{ra.get('take_profit_pct', 0):.1f}%
  Risk/Reward:      {ra.get('risk_reward_ratio', 0):.2f}×
  Warnings:         {'; '.join((ra.get('warnings') or [])[:3])}

PORTFOLIO CONTEXT:
  Total Value:      ${state.get('portfolio_value', 10000):,.0f}
  Risk Tolerance:   {state.get('risk_tolerance', 'moderate')}
  User Query:       {state.get('user_query', '')}
"""


def _extract_json(content: str) -> dict[str, Any]:
    """Extract the first JSON block from a model response."""
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fallback: try to parse the whole response
    return json.loads(content)


async def generate_hypotheses_node(state: GoTState) -> dict:
    """
    NODE 1 — Generate.

    STINKMEANER and SAMUEL each call the LLM independently and produce
    one ThoughtNode. Both calls happen in the same node (sequential here,
    but could be made truly parallel with asyncio.gather if desired).

    The generation number is stamped onto each node so we can reconstruct
    which loop iteration produced which thoughts in the audit trail.
    """
    llm      = _llm()
    gen      = state.get("got_generation", 1)
    data_ctx = _build_data_summary(state)
    new_thoughts: list[ThoughtNode] = []

    for persona, system_prompt in [
        ("STINKMEANER", STINKMEANER_SYSTEM),
        ("SAMUEL",      SAMUEL_SYSTEM),
    ]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate your hypothesis now.\n\n{data_ctx}"),
        ]
        try:
            response = llm.invoke(messages)
            content  = response.content if hasattr(response, "content") else str(response)
            parsed   = _extract_json(content)

            node = ThoughtNode(
                node_id           = str(uuid.uuid4()),
                parent_ids        = [],
                agent_origin      = persona,
                content           = parsed,
                confidence_score  = float(parsed.get("confidence_score", 0.7)),
                generation        = gen,
                critique_notes    = [],
                survived_critique = False,
            )
            new_thoughts.append(node)

        except Exception as e:
            # If the LLM call fails, add a low-confidence fallback node so
            # the graph can continue. The adversary will likely prune it.
            fallback = ThoughtNode(
                node_id           = str(uuid.uuid4()),
                parent_ids        = [],
                agent_origin      = persona,
                content           = {"thesis": f"FALLBACK — generation error: {e}",
                                     "direction": "neutral"},
                confidence_score  = 0.1,
                generation        = gen,
                critique_notes    = [f"Generation failed: {e}"],
                survived_critique = False,
            )
            new_thoughts.append(fallback)

    return {
        "thought_graph":    new_thoughts,
        "got_generation":   gen,
        "current_step":     f"hypotheses_generated_gen{gen}",
        "consensus_reached": False,
        "surviving_thoughts": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Adversarial Critique
# Persona: GRANDDAD (veteran risk sceptic with "PTSD" from past crashes)
# ─────────────────────────────────────────────────────────────────────────────

GRANDDAD_SYSTEM = """\
You are GRANDDAD — the most battle-scarred risk sceptic in this trading system.
You have survived the 2008 crash, the 2018 crypto winter, the COVID dump,
the Luna collapse, and the FTX contagion. You trust nothing.

Your job is to adversarially stress-test a trading hypothesis. For each
hypothesis you receive, apply ALL of the following attack scenarios and
assess whether the core trade thesis survives each one:

ATTACK CATALOGUE (apply all four to every hypothesis):
  1. FLASH_CRASH     — price drops 15-25% in under 60 minutes on no news.
                       Does the stop-loss protect capital? Is the position
                       size small enough to survive the wick?
  2. VOLUME_SPOOF    — the volume data driving the thesis is fabricated by
                       large players who will reverse the moment retail enters.
                       What does the thesis look like if volume drops 80%?
  3. MACRO_SHOCK     — a major macro event (Fed surprise, stablecoin depeg,
                       exchange insolvency) causes a 3-5% broad market move
                       against the position within 24 hours.
  4. LIQUIDITY_DRAIN — bid-side liquidity evaporates. The position cannot
                       be exited at the proposed stop price; actual exit
                       is 3-8% worse than the stop.

For each attack, decide:
  - Does the thesis still make sense?
  - By how much does your confidence in the thesis drop? (0.0 to 0.5 penalty)

Output ONLY a JSON array inside ```json ... ``` where each element is:
{
  "attack_type": "flash_crash" | "volume_spoof" | "macro_shock" | "liquidity_drain",
  "scenario": "specific description of this attack applied to THIS thesis",
  "confidence_delta": float  // negative number, e.g. -0.20 means 20% confidence reduction
  "thesis_survives": true | false,
  "explanation": "why the thesis does or does not survive this specific attack"
}
"""


async def adversarial_critique_node(state: GoTState) -> dict:
    """
    NODE 2 — Adversarial Critique.

    GRANDDAD attacks every thought in the current generation that hasn't
    yet been evaluated. For each thought, he runs all four stress tests and
    computes the total confidence penalty. If the post-attack confidence
    falls at or below SURVIVAL_THRESHOLD, the thought is pruned.

    All attacks — including attacks on pruned thoughts — are preserved in
    adversarial_attacks for the audit trail. This means the approval dashboard
    can show the human reviewer exactly what risks were considered and which
    ones the surviving thesis withstood.
    """
    llm = _llm()
    gen = state.get("got_generation", 1)

    # Only evaluate thoughts from the current generation that haven't been
    # through the critique yet. This prevents re-attacking thoughts from
    # prior loop iterations.
    current_gen_thoughts = [
        t for t in state.get("thought_graph", [])
        if t["generation"] == gen and not t["survived_critique"]
    ]

    updated_thoughts: list[ThoughtNode]  = []
    all_attacks:      list[AdversarialAttack] = []
    survivors:        list[ThoughtNode]  = []

    for thought in current_gen_thoughts:
        # Build a concise description of this hypothesis for GRANDDAD
        content = thought["content"]
        hypothesis_text = (
            f"Hypothesis from {thought['agent_origin']}:\n"
            f"  Thesis:       {content.get('thesis', '?')}\n"
            f"  Direction:    {content.get('direction', '?')}\n"
            f"  Timeframe:    {content.get('timeframe', '?')}\n"
            f"  Entry:        ${content.get('proposed_entry', 0):,.4f}\n"
            f"  Target:       ${content.get('proposed_target', 0):,.4f}\n"
            f"  Stop:         ${content.get('proposed_stop', 0):,.4f}\n"
            f"  Evidence:     {'; '.join(content.get('key_evidence', []))}\n"
            f"  Primary Risk: {content.get('primary_risk', '?')}\n"
            f"  Confidence:   {thought['confidence_score']:.0%}\n"
            f"\nCurrent market price: ${state.get('market_data', {}).get('price_usd', 0):,.4f}\n"
            f"Portfolio value: ${state.get('portfolio_value', 10000):,.0f}\n"
            f"Risk tolerance: {state.get('risk_tolerance', 'moderate')}"
        )

        messages = [
            SystemMessage(content=GRANDDAD_SYSTEM),
            HumanMessage(content=f"Apply all four attacks to this hypothesis:\n\n{hypothesis_text}"),
        ]

        try:
            response    = llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)
            attacks_raw = _extract_json(raw_content)
            # The model returns a list, not a dict
            if not isinstance(attacks_raw, list):
                attacks_raw = [attacks_raw]
        except Exception as e:
            # If GRANDDAD fails, apply a conservative penalty and continue
            attacks_raw = [{
                "attack_type":      "system_error",
                "scenario":         f"Critique node error: {e}",
                "confidence_delta": -0.20,
                "thesis_survives":  True,
                "explanation":      "Critique failed; applying conservative penalty.",
            }]

        # Compute cumulative confidence after all attacks
        total_delta    = sum(float(a.get("confidence_delta", 0)) for a in attacks_raw)
        post_attack_cf = thought["confidence_score"] + total_delta   # delta is negative
        post_attack_cf = max(0.0, min(1.0, post_attack_cf))

        critique_notes = [
            f"[{a.get('attack_type','?').upper()}] "
            f"{'✓ survived' if a.get('thesis_survives') else '✗ failed'} — "
            f"{a.get('explanation','')[:120]}"
            for a in attacks_raw
        ]

        # Build AdversarialAttack records for the audit trail
        for a in attacks_raw:
            all_attacks.append(AdversarialAttack(
                attack_id       = str(uuid.uuid4()),
                target_node_id  = thought["node_id"],
                attack_type     = a.get("attack_type", "unknown"),
                scenario        = a.get("scenario", ""),
                confidence_delta= float(a.get("confidence_delta", 0)),
                thesis_survives = bool(a.get("thesis_survives", False)),
                explanation     = a.get("explanation", ""),
            ))

        # Create an updated ThoughtNode with critique results stamped in
        updated = ThoughtNode(
            node_id           = thought["node_id"],
            parent_ids        = thought["parent_ids"],
            agent_origin      = thought["agent_origin"],
            content           = thought["content"],
            confidence_score  = post_attack_cf,
            generation        = thought["generation"],
            critique_notes    = critique_notes,
            survived_critique = True,  # evaluated regardless of whether it passed
        )
        updated_thoughts.append(updated)

        if post_attack_cf > SURVIVAL_THRESHOLD:
            survivors.append(updated)

    # Replace the old (uncritiqued) versions of these thoughts in the graph
    # with the updated (critiqued) versions. We do this by returning them
    # through the thought_graph field — merge_thoughts deduplicates by node_id,
    # but because we're using the SAME node_ids, we need to patch the existing
    # entries rather than append new ones.
    # Strategy: return the full updated list as surviving_thoughts and let
    # the graph re-read from there. The merge_thoughts reducer handles dedup.

    return {
        "thought_graph":       updated_thoughts,   # reducer merges by node_id awareness
        "adversarial_attacks": all_attacks,
        "surviving_thoughts":  survivors,
        "current_step":        f"adversarial_critique_complete_gen{gen}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Aggregate and Improve
# Personas: CLAYTON (risk) and JULIUS (execution pragmatism)
# ─────────────────────────────────────────────────────────────────────────────

CLAYTON_JULIUS_SYSTEM = """\
You are the synthesis team: CLAYTON (the defensive risk officer) and
JULIUS (the pragmatic execution closer). Together you receive the surviving
trading hypotheses — the ones that withstood GRANDDAD's adversarial
stress tests — and produce a single, unified, risk-adjusted trade recommendation.

Your job is not to pick a winner between the surviving hypotheses. It is to
MERGE them: find where they agree, reconcile where they differ, and produce
a recommendation that is more conservative than either hypothesis alone.

Rules:
  1. ENTRY ZONE: If both hypotheses agree on direction, the entry zone spans
     from the more conservative entry to the more aggressive. If they disagree
     on direction, recommend HOLD and explain clearly.
  2. TARGET: Take the average of proposed targets, then reduce by 10% for
     conservatism (CLAYTON's influence).
  3. STOP-LOSS: Take the tighter (closer to entry) of the two proposed stops,
     then tighten it by a further 5% (capital preservation first).
  4. POSITION SIZE: Apply the risk_assessment's max_position_size_pct directly.
     Never recommend more than the Risk Manager approved.
  5. CONFIDENCE: The merged confidence is the weighted average of the surviving
     thoughts' post-attack confidence scores, weighted by their generation
     (newer generations are weighted slightly higher as they had more context).

Your final output must include the full reasoning for each decision — this
rationale goes directly to the human reviewer in the approval dashboard.

Output ONLY a JSON object inside ```json ... ```:
{
  "action": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell" | "avoid",
  "entry_zone": {"low": float, "high": float},
  "target_price": float,
  "stop_loss": float,
  "timeframe": "1d" | "3d" | "1w" | "2w",
  "confidence": float,
  "strategy_used": string,
  "rationale": "detailed 4-6 sentence rationale referencing both hypotheses and the attacks they survived",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "dca_plan": "optional DCA approach if applicable, otherwise empty string",
  "synthesis_notes": {
    "hypotheses_merged": int,
    "attacks_survived":  int,
    "hypotheses_pruned": int,
    "dominant_thesis":   "STINKMEANER" | "SAMUEL" | "balanced"
  }
}
"""


async def aggregate_and_improve_node(state: GoTState) -> dict:
    """
    NODE 3 — Aggregate and Improve.

    CLAYTON and JULIUS receive the surviving thoughts and the full audit of
    adversarial attacks, then produce a merged recommendation that is more
    conservative and better-documented than either hypothesis alone.

    If no thoughts survived (called when got_max_generations is reached),
    this node produces a HOLD recommendation with an honest rationale rather
    than fabricating conviction the system doesn't have.
    """
    llm        = _llm()
    survivors  = state.get("surviving_thoughts", [])
    all_attacks = state.get("adversarial_attacks", [])
    gen        = state.get("got_generation", 1)
    max_gen    = state.get("got_max_generations", 3)

    # ── No survivors edge case ────────────────────────────────────────────────
    if not survivors:
        pruned_count  = len([t for t in state.get("thought_graph", [])])
        hold_rec: TradeRecommendation = {
            "action":       "hold",
            "entry_zone":   {"low": 0.0, "high": 0.0},
            "target_price": 0.0,
            "stop_loss":    0.0,
            "timeframe":    "1w",
            "confidence":   0.1,
            "strategy_used": "adversarial_pruning",
            "rationale":    (
                f"All {pruned_count} hypotheses across {gen} generation(s) "
                f"failed the adversarial stress tests applied by GRANDDAD. "
                f"The system cannot produce a directional recommendation with "
                f"sufficient confidence under current market conditions. "
                f"HOLD until conditions clarify. Do not force a trade."
            ),
            "key_risks":   ["All directional theses failed stress testing",
                            "Market conditions currently unfavourable for conviction",
                            "Retry after next major price/volume event"],
            "dca_plan":    "",
        }
        return {
            "trade_recommendation": hold_rec,
            "consensus_reached":    True,
            "current_step":         "aggregation_complete_no_survivors",
        }

    # ── Build synthesis prompt ────────────────────────────────────────────────
    # Summarise each surviving thought for CLAYTON/JULIUS
    survivor_summaries = []
    for t in survivors:
        c = t["content"]
        attacks_on_this = [a for a in all_attacks if a["target_node_id"] == t["node_id"]]
        survived_count  = sum(1 for a in attacks_on_this if a["thesis_survives"])
        survivor_summaries.append(
            f"\n{'─'*50}\n"
            f"Origin: {t['agent_origin']} | Post-attack confidence: {t['confidence_score']:.0%} | "
            f"Generation: {t['generation']}\n"
            f"Survived {survived_count}/{len(attacks_on_this)} attacks\n"
            f"Thesis:     {c.get('thesis', '?')}\n"
            f"Direction:  {c.get('direction', '?')}\n"
            f"Timeframe:  {c.get('timeframe', '?')}\n"
            f"Entry:      ${c.get('proposed_entry', 0):,.4f}\n"
            f"Target:     ${c.get('proposed_target', 0):,.4f}\n"
            f"Stop:       ${c.get('proposed_stop', 0):,.4f}\n"
            f"Evidence:   {'; '.join(c.get('key_evidence', []))}\n"
            f"Critique:   {chr(10).join(t.get('critique_notes', []))}"
        )

    pruned_count = len(state.get("thought_graph", [])) - len(survivors)
    ra = state.get("risk_assessment", {})

    synthesis_prompt = (
        f"Synthesise these {len(survivors)} surviving hypothesis/hypotheses "
        f"into a unified recommendation.\n"
        f"{len(pruned_count if isinstance(pruned_count, list) else range(pruned_count))} "
        f"hypothesis/hypotheses were pruned by the adversary.\n\n"
        f"SURVIVING HYPOTHESES:\n{''.join(survivor_summaries)}\n\n"
        f"RISK CONSTRAINTS (from Risk Manager — do not exceed these):\n"
        f"  Max position: {ra.get('max_position_size_pct', 20):.1f}% "
        f"(${ra.get('max_position_size_usd', 2000):,.0f})\n"
        f"  Risk level:   {ra.get('risk_level', 'medium')}\n\n"
        f"Total attacks survived across all hypotheses: "
        f"{sum(1 for a in all_attacks if a['thesis_survives'])}/{len(all_attacks)}"
    )

    messages = [
        SystemMessage(content=CLAYTON_JULIUS_SYSTEM),
        HumanMessage(content=synthesis_prompt),
    ]

    try:
        response = llm.invoke(messages)
        content  = response.content if hasattr(response, "content") else str(response)
        parsed   = _extract_json(content)

        # Safety: cap confidence at what the post-attack scores support
        avg_survivor_cf = sum(t["confidence_score"] for t in survivors) / len(survivors)
        parsed["confidence"] = min(
            float(parsed.get("confidence", avg_survivor_cf)),
            avg_survivor_cf + 0.05,  # allow tiny uplift from synthesis, no more
        )

        # Enrich rationale with audit context
        attacks_survived = sum(1 for a in all_attacks if a["thesis_survives"])
        parsed["rationale"] = (
            f"[GoT: {len(survivors)} hypothesis survived {gen} generation(s), "
            f"{attacks_survived}/{len(all_attacks)} stress tests passed] "
            + parsed.get("rationale", "")
        )

        return {
            "trade_recommendation": parsed,
            "consensus_reached":    True,
            "current_step":         "aggregation_complete",
        }

    except Exception as e:
        # Graceful degradation: fall back to the highest-confidence survivor
        best = max(survivors, key=lambda t: t["confidence_score"])
        c    = best["content"]
        fallback_rec = {
            "action":       _direction_to_action(c.get("direction", "neutral"),
                                                  best["confidence_score"]),
            "entry_zone":   {"low": float(c.get("proposed_entry", 0)) * 0.99,
                             "high": float(c.get("proposed_entry", 0)) * 1.01},
            "target_price": float(c.get("proposed_target", 0)),
            "stop_loss":    float(c.get("proposed_stop", 0)),
            "timeframe":    c.get("timeframe", "1w"),
            "confidence":   best["confidence_score"],
            "strategy_used": "got_fallback",
            "rationale":    (
                f"Aggregation synthesis failed ({e}); falling back to "
                f"highest-confidence surviving hypothesis from {best['agent_origin']}. "
                f"Thesis: {c.get('thesis', '?')}"
            ),
            "key_risks":   c.get("key_risks", []),
            "dca_plan":    "",
        }
        return {
            "trade_recommendation": fallback_rec,
            "consensus_reached":    True,
            "current_step":         "aggregation_complete_fallback",
        }


def _direction_to_action(direction: str, confidence: float) -> str:
    """Map a direction string and confidence to a trade action."""
    if direction == "long":
        return "strong_buy" if confidence > 0.80 else "buy"
    if direction == "short":
        return "strong_sell" if confidence > 0.80 else "sell"
    return "hold"


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge function
# ─────────────────────────────────────────────────────────────────────────────

def route_after_critique(state: GoTState) -> str:
    """
    After the adversarial critique, decide whether to:
      "aggregate"  — survivors exist; proceed to final synthesis.
      "generate"   — all hypotheses were pruned; retry with a new generation.
      "aggregate"  — max generations reached; force aggregation (will produce HOLD).

    This function is the heart of the GoT loop. It embodies the principle
    that the system would rather say "I don't know" honestly than force a
    low-confidence recommendation through to the approval queue.
    """
    survivors  = state.get("surviving_thoughts", [])
    gen        = state.get("got_generation", 1)
    max_gen    = state.get("got_max_generations", 3)

    if survivors:
        # We have at least one hypothesis that survived stress testing
        return "aggregate"

    if gen >= max_gen:
        # Hit the retry ceiling — route to aggregate which will produce HOLD
        return "aggregate"

    # No survivors and retries remaining — loop back to generate
    return "generate"


def increment_generation(state: GoTState) -> dict:
    """
    A lightweight node that increments got_generation before the graph
    loops back to generate_hypotheses. LangGraph needs this as an
    explicit node rather than side-effecting the conditional edge.
    """
    return {
        "got_generation": state.get("got_generation", 1) + 1,
        "current_step":   f"retrying_generation_{state.get('got_generation', 1) + 1}",
    }
