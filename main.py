"""
main.py — Interactive CLI for the Crypto Multi-Agent System.

Usage:
    python main.py

Or with arguments:
    python main.py --symbol ETH --portfolio 50000 --risk moderate
    python main.py --query "Should I buy BTC now?" --portfolio 10000
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text

load_dotenv()

console = Console()

BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║           🚀  CRYPTO MULTI-AGENT TRADING SYSTEM  🚀              ║
║                  Powered by Claude + LangGraph                   ║
║                                                                   ║
║  Agents: Orchestrator · Market Analyst · Technical Analyst       ║
║          Risk Manager · Strategy Advisor                         ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def check_env():
    """Validate required environment variables."""
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        console.print(
            f"[red]❌ Missing required env vars: {', '.join(missing)}[/red]\n"
            "Copy .env.example → .env and fill in your keys."
        )
        sys.exit(1)


def get_initial_state(
    query: str,
    symbol: str = "BTC",
    portfolio: float = 10_000.0,
    risk: str = "moderate",
    holdings: dict | None = None,
    memory_context: str = "",
) -> dict:
    """Build the initial AgentState dict."""
    return {
        "messages":         [],
        "user_query":       query,
        "target_symbol":    symbol.upper(),
        "portfolio_value":  portfolio,
        "current_holdings": holdings or {},
        "risk_tolerance":   risk,
        "active_agents":    [],
        "current_step":     "initialising",
        "iteration":        0,
        "market_data":      {},
        "technical_analysis": {},
        "risk_assessment":  {},
        "trade_recommendation": {},
        "memory_context":   memory_context,
        "final_response":   "",
        "errors":           [],
    }


def print_progress(step: str):
    step_labels = {
        "routing_complete":          "🧭 Routing query to specialists...",
        "market_analysis_complete":  "📊 Market analysis done",
        "technical_analysis_complete": "📈 Technical analysis done",
        "risk_assessment_complete":  "⚖️  Risk assessment done",
        "strategy_complete":         "🎯 Strategy formulated",
        "synthesis_complete":        "✅ Synthesising final response...",
    }
    label = step_labels.get(step, f"⚙️  {step}")
    console.print(f"  [dim]{label}[/dim]")


def run_query(
    graph,
    memory_manager,
    query: str,
    symbol: str,
    portfolio: float,
    risk: str,
    session_id: str,
    holdings: dict | None = None,
):
    """Run a single query through the agent graph."""
    # Load persistent memory
    memory_context = memory_manager.build_context(symbol)

    initial_state = get_initial_state(
        query=query,
        symbol=symbol,
        portfolio=portfolio,
        risk=risk,
        holdings=holdings,
        memory_context=memory_context,
    )

    config = {"configurable": {"thread_id": session_id}}

    console.print(Rule(f"[bold cyan]Query: {query}[/bold cyan]"))
    console.print()

    final_state = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running multi-agent analysis...", total=None)

        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            step = event.get("current_step", "")
            if step:
                progress.update(task, description=step.replace("_", " ").title())
            final_state = event

    if not final_state:
        console.print("[red]No response generated.[/red]")
        return

    # Print the response
    response = final_state.get("final_response", "")
    if response:
        console.print(
            Panel(
                Markdown(response),
                title="[bold green]🤖 Agent Analysis[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        console.print("[yellow]No final response in state.[/yellow]")

    # Save summary to memory
    summary = (
        f"Analysed {final_state.get('target_symbol', symbol)} | "
        f"Query: {query[:100]} | "
        f"Recommendation: {final_state.get('trade_recommendation', {}).get('action', 'N/A')}"
    )
    memory_manager.save_summary(session_id, summary)

    return final_state


def interactive_session(symbol: str, portfolio: float, risk: str, graph, memory_manager):
    """Run an interactive REPL session."""
    session_id = str(uuid.uuid4())[:8]
    console.print(f"\n[dim]Session ID: {session_id}[/dim]")
    console.print(
        f"[bold]Portfolio:[/bold] ${portfolio:,.0f} | "
        f"[bold]Risk:[/bold] {risk} | "
        f"[bold]Default symbol:[/bold] {symbol}"
    )
    console.print(
        "\n[dim]Commands: 'quit' to exit | 'symbol BTC' to change coin | "
        "'portfolio 50000' to update | 'history' to see trades[/dim]\n"
    )

    holdings: dict = {}

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        if not user_input:
            continue

        # Meta commands
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! 👋[/dim]")
            break
        elif user_input.lower().startswith("symbol "):
            symbol = user_input.split()[1].upper()
            console.print(f"[green]Symbol changed to {symbol}[/green]")
            continue
        elif user_input.lower().startswith("portfolio "):
            try:
                portfolio = float(user_input.split()[1].replace(",", ""))
                console.print(f"[green]Portfolio updated to ${portfolio:,.0f}[/green]")
            except ValueError:
                console.print("[red]Invalid portfolio value[/red]")
            continue
        elif user_input.lower().startswith("risk "):
            risk_val = user_input.split()[1].lower()
            if risk_val in ("conservative", "moderate", "aggressive"):
                risk = risk_val
                console.print(f"[green]Risk tolerance set to {risk}[/green]")
            else:
                console.print("[red]Use: conservative | moderate | aggressive[/red]")
            continue
        elif user_input.lower() == "history":
            stats = memory_manager.get_trade_stats()
            trades = memory_manager.get_recent_trades(5)
            console.print(Panel(
                f"Total trades: {stats.get('total_trades', 0)}\n"
                f"Win rate: {stats.get('win_rate_pct', 0)}%\n"
                f"Total P&L: ${stats.get('total_pnl_usd', 0):,.2f}\n\n"
                + "\n".join(
                    f"{t['symbol']} {t['action']} @ ${t['entry_price']} | "
                    f"{'$' + str(t['pnl_usd']) if t['pnl_usd'] else 'OPEN'}"
                    for t in trades
                ),
                title="Trade History",
            ))
            continue
        elif user_input.lower() == "memory":
            ctx = memory_manager.build_context(symbol)
            console.print(Panel(ctx, title="Memory Context"))
            continue

        # Run through the agent graph
        run_query(
            graph=graph,
            memory_manager=memory_manager,
            query=user_input,
            symbol=symbol,
            portfolio=portfolio,
            risk=risk,
            session_id=session_id,
            holdings=holdings,
        )
        console.print()


def main():
    check_env()

    parser = argparse.ArgumentParser(description="Crypto Multi-Agent Trading System")
    parser.add_argument("--symbol",    default="BTC",       help="Default crypto symbol")
    parser.add_argument("--portfolio", default=10000.0, type=float, help="Portfolio value in USD")
    parser.add_argument("--risk",      default="moderate",
                        choices=["conservative", "moderate", "aggressive"])
    parser.add_argument("--query",     help="Run a single query (non-interactive)")
    parser.add_argument("--db",        default="./memory/langgraph.db",
                        help="LangGraph checkpoint DB path")
    parser.add_argument("--memory-db", default="./memory/agent_memory.db",
                        help="Custom memory DB path")
    args = parser.parse_args()

    console.print(f"[bold magenta]{BANNER}[/bold magenta]")

    # Lazy imports (after env check)
    from graph import build_graph
    from memory.memory_manager import CryptoMemoryManager

    console.print("[dim]Initialising agents...[/dim]")
    graph = build_graph(args.db)
    memory_manager = CryptoMemoryManager(args.memory_db)
    console.print("[green]✓ System ready[/green]\n")

    if args.query:
        # Single-shot mode
        run_query(
            graph=graph,
            memory_manager=memory_manager,
            query=args.query,
            symbol=args.symbol,
            portfolio=args.portfolio,
            risk=args.risk,
            session_id=str(uuid.uuid4())[:8],
        )
    else:
        # Interactive REPL
        interactive_session(
            symbol=args.symbol,
            portfolio=args.portfolio,
            risk=args.risk,
            graph=graph,
            memory_manager=memory_manager,
        )


if __name__ == "__main__":
    main()
