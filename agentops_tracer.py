"""
observability/agentops_tracer.py
─────────────────────────────────
AgentOps integration for complete LangGraph trace visibility.

WHAT THIS GIVES YOU:
  Every node invocation in the LangGraph graph — orchestrator, all four
  specialists, every GoT node, the approval workflow — is traced with:
    • Wall-clock timing per node
    • Input and output payloads (truncated for token efficiency)
    • Error traces with full stack context
    • Token usage per LLM call
    • Session-level grouping (one "session" per analysis run)

  All traces appear in your AgentOps dashboard at https://app.agentops.ai
  The Prometheus MCP server can then query these traces conversationally.

HOW TO USE:
  1. pip install agentops
  2. Set AGENTOPS_API_KEY env var
  3. In server.py lifespan startup, call: init_agentops()
  4. All LangGraph node calls are automatically traced via the
     HydraLangGraphTracer callback handler.

MANUAL SPAN USAGE (inside any node function):
  with trace_node("my_node_name", {"input_key": "value"}) as span:
      result = do_work()
      span.add_output(result)

VIEWING TRACES:
  - agentops.ai dashboard shows all sessions with timelines
  - The Prometheus MCP tool query_metrics can pull aggregated
    latency/error stats from the Prometheus sidecar that AgentOps
    data feeds into via the prometheus_metrics module.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

AGENTOPS_KEY  = os.getenv("AGENTOPS_API_KEY", "")
ENABLED       = bool(AGENTOPS_KEY)

# ── Optional import ───────────────────────────────────────────────────────────
try:
    import agentops
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False
    logger.info("agentops not installed — tracing disabled. pip install agentops")


# ── Initialisation ────────────────────────────────────────────────────────────

def init_agentops(
    api_key:     str  = AGENTOPS_KEY,
    tags:        list[str] | None = None,
    auto_start:  bool = True,
) -> bool:
    """
    Initialise the AgentOps SDK. Call once at server startup.
    Returns True if initialisation succeeded, False otherwise.
    """
    if not AGENTOPS_AVAILABLE or not api_key:
        logger.warning("AgentOps not available or API key missing — tracing disabled")
        return False

    try:
        agentops.init(
            api_key=api_key,
            tags=tags or ["hydra-sentinel-x", "crypto-trading", "got-reasoning"],
            auto_start_session=auto_start,
        )
        logger.info("AgentOps initialised — tracing active")
        return True
    except Exception as e:
        logger.error(f"AgentOps init failed: {e}")
        return False


# ── Session management ────────────────────────────────────────────────────────

class HydraSession:
    """
    Represents one analysis session in AgentOps — from user query to
    final recommendation delivered to the approval dashboard.

    Each session groups all node traces so you can see the complete
    reasoning chain for one analysis run in the dashboard.
    """

    def __init__(self, session_id: str, symbol: str, user_query: str):
        self.session_id = session_id
        self.symbol     = symbol
        self.query      = user_query
        self._session   = None
        self._start     = time.time()
        self._node_times: dict[str, float] = {}

    def start(self) -> None:
        if not AGENTOPS_AVAILABLE or not ENABLED:
            return
        try:
            self._session = agentops.start_session(
                tags=[
                    f"symbol:{self.symbol}",
                    f"session:{self.session_id}",
                ]
            )
        except Exception as e:
            logger.debug(f"AgentOps session start: {e}")

    def end(self, success: bool = True) -> None:
        if not AGENTOPS_AVAILABLE or not ENABLED or not self._session:
            return
        try:
            duration = round(time.time() - self._start, 2)
            agentops.end_session(
                end_state="Success" if success else "Fail",
                end_state_reason=f"Analysis completed in {duration}s",
            )
        except Exception as e:
            logger.debug(f"AgentOps session end: {e}")

    def record_node(
        self,
        node_name:    str,
        duration_ms:  float,
        input_summary: str,
        output_summary: str,
        error: str | None = None,
    ) -> None:
        """Record a single LangGraph node execution."""
        self._node_times[node_name] = duration_ms
        if not AGENTOPS_AVAILABLE or not ENABLED:
            return
        try:
            agentops.record(agentops.ActionEvent(
                action_type=f"langgraph_node:{node_name}",
                params={
                    "node":        node_name,
                    "session_id":  self.session_id,
                    "symbol":      self.symbol,
                    "input":       input_summary[:300],
                    "duration_ms": round(duration_ms, 1),
                },
                returns=output_summary[:300] if not error else f"ERROR: {error}",
                init_timestamp=None,
                end_timestamp=None,
            ))
        except Exception as e:
            logger.debug(f"AgentOps record_node: {e}")

    def timing_summary(self) -> dict[str, float]:
        return dict(self._node_times)


# ── Node tracing decorator ────────────────────────────────────────────────────

def traced_node(
    node_name: str | None = None,
    session_extractor: Callable | None = None,
):
    """
    Decorator for LangGraph node functions. Automatically records
    execution timing and input/output summaries.

    Usage:
        @traced_node("market_analyst")
        def market_analyst_node(state: AgentState) -> dict:
            ...

    The decorator does NOT alter the function's return value — it only
    records timing and payloads as side effects.

    If AgentOps is not available, the decorator is a transparent passthrough.
    """
    def decorator(fn: Callable) -> Callable:
        name = node_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            t_start = time.perf_counter()
            error   = None
            result  = None

            try:
                result = fn(state, *args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - t_start) * 1000

                # Build terse summaries — don't log full state
                input_summary = (
                    f"symbol={state.get('target_symbol','?')} "
                    f"step={state.get('current_step','?')} "
                    f"portfolio=${state.get('portfolio_value',0):,.0f}"
                )
                output_summary = (
                    f"keys_written={list(result.keys()) if result else '[]'}"
                    if not error else f"error={error}"
                )

                logger.debug(
                    f"[TRACE] {name} | {duration_ms:.0f}ms | "
                    f"{output_summary}"
                )

                # Record to AgentOps if available
                if AGENTOPS_AVAILABLE and ENABLED:
                    try:
                        agentops.record(agentops.ActionEvent(
                            action_type=f"node:{name}",
                            params={"input": input_summary, "duration_ms": round(duration_ms, 1)},
                            returns=output_summary,
                        ))
                    except Exception:
                        pass

        return wrapper
    return decorator


# ── Context manager for manual spans ─────────────────────────────────────────

@contextmanager
def trace_span(
    name:    str,
    payload: dict[str, Any] | None = None,
) -> Generator[dict, None, None]:
    """
    Context manager for manually tracing a code block within a node.

    with trace_span("adversarial_critique", {"hypothesis": "BTC long"}) as span:
        result = run_critique()
        span["output"] = str(result)
    """
    span: dict[str, Any] = {
        "name":    name,
        "input":   payload or {},
        "output":  None,
        "error":   None,
        "start":   time.perf_counter(),
    }
    try:
        yield span
    except Exception as e:
        span["error"] = str(e)
        raise
    finally:
        span["duration_ms"] = round((time.perf_counter() - span["start"]) * 1000, 1)
        logger.debug(
            f"[SPAN] {name} | {span['duration_ms']}ms | "
            f"{'OK' if not span['error'] else 'ERR:' + span['error']}"
        )
        if AGENTOPS_AVAILABLE and ENABLED:
            try:
                agentops.record(agentops.ActionEvent(
                    action_type=f"span:{name}",
                    params={"input": str(span["input"])[:200]},
                    returns=str(span.get("output", ""))[:200]
                          if not span["error"]
                          else f"ERROR: {span['error']}",
                ))
            except Exception:
                pass


# ── LangChain callback handler ────────────────────────────────────────────────

class HydraLangGraphTracer:
    """
    LangChain callback handler that instruments all LLM calls made
    by the specialist agents and GoT nodes.

    Attach to ChatAnthropic via:
        llm = ChatAnthropic(..., callbacks=[HydraLangGraphTracer(session_id)])

    Records:
      - LLM call start / end times
      - Token usage (input_tokens, output_tokens)
      - Agent name (extracted from the system message prefix)
    """

    def __init__(self, session_id: str = "", symbol: str = ""):
        self.session_id = session_id
        self.symbol     = symbol
        self._pending:  dict[str, float] = {}   # run_id → start_time

    def on_llm_start(self, serialized: dict, prompts: list, run_id: str = "", **kwargs) -> None:
        self._pending[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response, run_id: str = "", **kwargs) -> None:
        start = self._pending.pop(str(run_id), None)
        if start is None:
            return

        duration_ms = (time.perf_counter() - start) * 1000
        usage       = {}
        try:
            usage = response.llm_output.get("usage", {}) if hasattr(response, "llm_output") else {}
        except Exception:
            pass

        logger.debug(
            f"[LLM] {duration_ms:.0f}ms | "
            f"in={usage.get('input_tokens','?')} "
            f"out={usage.get('output_tokens','?')}"
        )

        if AGENTOPS_AVAILABLE and ENABLED:
            try:
                agentops.record(agentops.LLMEvent(
                    prompt=f"[{self.symbol}] LLM call",
                    completion="[truncated]",
                    model="claude-sonnet-4-6",
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                ))
            except Exception:
                pass

    def on_llm_error(self, error: Exception, run_id: str = "", **kwargs) -> None:
        self._pending.pop(str(run_id), None)
        logger.error(f"[LLM ERROR] {error}")
        if AGENTOPS_AVAILABLE and ENABLED:
            try:
                agentops.record(agentops.ErrorEvent(
                    trigger_event=agentops.ActionEvent(action_type="llm_error"),
                    exception=error,
                ))
            except Exception:
                pass
