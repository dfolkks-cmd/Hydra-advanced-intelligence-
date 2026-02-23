"""
cdp/wallet_manager.py — Coinbase Developer Platform (CDP) wallet integration.

This module solves three problems the basic snippet doesn't:

  1. PERSISTENCE   — Wallets are created once and reloaded by ID every session.
                     Without this, you'd create a new empty wallet every run.

  2. SAFETY LAYER  — Every transaction is checked against Risk Manager limits
                     before execution. The agent cannot move more than the
                     configured max_trade_pct of the portfolio in one call.

  3. TOOL BINDING  — Every action is wrapped as a @tool so LangGraph agents
                     can call them natively, just like get_crypto_price().

WALLET PERSISTENCE EXPLAINED:
  Coinbase MPC wallets are non-custodial: Coinbase holds one key share,
  your server holds another. Neither party alone can sign a transaction.
  The wallet's unique ID is stored locally in a JSON file (wallet_state.json).
  On every startup, we load that file and call Wallet.fetch() to reconnect
  to the *same* wallet — not create a new one.

SETUP:
  1. pip install cdp-sdk
  2. Place your cdp_api_key.json somewhere safe (e.g. ~/.cdp/cdp_api_key.json)
  3. Set CDP_KEY_PATH in your .env file
  4. Run: python -m cdp.wallet_manager --init   (creates your wallet once)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ── Try importing the CDP SDK (gracefully degrade if not installed) ──────────
try:
    from cdp import Cdp, Wallet
    from cdp.errors import ApiError
    CDP_AVAILABLE = True
except ImportError:
    CDP_AVAILABLE = False
    # Define stubs so the rest of the file parses cleanly
    class Wallet:   # type: ignore
        pass
    class ApiError(Exception):   # type: ignore
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants (loaded from .env)
# ─────────────────────────────────────────────────────────────────────────────

CDP_KEY_PATH      = os.getenv("CDP_KEY_PATH", "~/.cdp/cdp_api_key.json")
WALLET_STATE_PATH = os.getenv("WALLET_STATE_PATH", "./cdp/wallet_state.json")

# Safety limit: no single agent-initiated transfer can exceed this % of
# the portfolio value stored in the risk profile. Default 5% = conservative.
MAX_TRANSFER_PCT  = float(os.getenv("CDP_MAX_TRANSFER_PCT", "5.0"))

# The blockchain network to operate on.
# "base-mainnet" = real money | "base-sepolia" = testnet (free fake ETH)
NETWORK_ID        = os.getenv("CDP_NETWORK_ID", "base-sepolia")


# ─────────────────────────────────────────────────────────────────────────────
# WalletManager — core class
# ─────────────────────────────────────────────────────────────────────────────

class WalletManager:
    """
    Manages a single persistent CDP MPC wallet for the trading agent system.

    The wallet is created once via .create_and_save(), then reloaded on every
    subsequent startup via .load(). This mirrors how a real trading desk
    maintains a single operational wallet rather than spinning up new ones.

    Example:
        manager = WalletManager()
        manager.load()  # reconnects to your existing wallet

        balance = manager.get_balance("eth")
        print(f"ETH available: {balance}")
    """

    def __init__(self):
        self._wallet: Optional[object] = None   # the live CDP Wallet object
        self._wallet_id: Optional[str] = None
        self._address: Optional[str] = None
        self._network: str = NETWORK_ID
        self._state_path = Path(WALLET_STATE_PATH).expanduser()

        # Auto-configure CDP SDK if the key file exists
        if CDP_AVAILABLE:
            key_path = Path(CDP_KEY_PATH).expanduser()
            if key_path.exists():
                Cdp.configure_from_json(str(key_path))
            else:
                raise FileNotFoundError(
                    f"CDP key file not found at: {key_path}\n"
                    f"Set CDP_KEY_PATH in your .env to point to cdp_api_key.json"
                )
        else:
            raise ImportError(
                "CDP SDK not installed. Run: pip install cdp-sdk\n"
                "Then add CDP_KEY_PATH to your .env file."
            )

    # ── Wallet lifecycle ─────────────────────────────────────────────────────

    def create_and_save(self, network_id: str = NETWORK_ID) -> str:
        """
        Create a brand-new MPC wallet and save its ID to disk.

        Call this ONCE when setting up the system for the first time.
        After this, always use .load() — calling create_and_save() again
        will create a DIFFERENT wallet and you'll lose track of the old one.

        Returns the new wallet's deposit address.
        """
        if self._state_path.exists():
            raise RuntimeError(
                f"Wallet state already exists at {self._state_path}.\n"
                "To create a new wallet, delete that file first (⚠️ only if intentional)."
            )

        print(f"Creating new CDP wallet on {network_id}...")
        wallet = Wallet.create(network_id=network_id)
        address = wallet.default_address

        # Persist the wallet ID so we can reload it in future sessions
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "wallet_id":  wallet.id,
            "network_id": network_id,
            "address":    str(address),
            "created_at": datetime.utcnow().isoformat(),
        }
        self._state_path.write_text(json.dumps(state, indent=2))

        self._wallet    = wallet
        self._wallet_id = wallet.id
        self._address   = str(address)
        self._network   = network_id

        print(f"✅ Wallet created and saved.")
        print(f"   Wallet ID : {wallet.id}")
        print(f"   Address   : {address}")
        print(f"   Network   : {network_id}")
        print(f"\n⚠️  Fund this address with test ETH before trading:")
        if "sepolia" in network_id:
            print(f"   https://www.coinbase.com/faucets/base-ethereum-goerli-faucet")
        return str(address)

    def load(self) -> str:
        """
        Reload an existing wallet from the saved state file.

        This is what you call on every application startup after the
        first-time setup. It reconnects to the same wallet without
        creating a new one.

        Returns the wallet's deposit address.
        """
        if not self._state_path.exists():
            raise FileNotFoundError(
                f"No wallet state found at {self._state_path}.\n"
                "Run: python -m cdp.wallet_manager --init to create one first."
            )

        state = json.loads(self._state_path.read_text())
        self._wallet_id = state["wallet_id"]
        self._address   = state["address"]
        self._network   = state.get("network_id", NETWORK_ID)

        # Fetch reconnects the Python object to the existing on-chain wallet
        self._wallet = Wallet.fetch(self._wallet_id)

        print(f"✅ Wallet loaded: {self._address} on {self._network}")
        return self._address

    def is_loaded(self) -> bool:
        return self._wallet is not None

    # ── Balance & Info ───────────────────────────────────────────────────────

    def get_balance(self, asset: str = "eth") -> float:
        """
        Return the current balance of an asset as a float.

        The asset string can be 'eth', 'usdc', 'cbeth', etc.
        On Base Sepolia (testnet), only 'eth' is typically available.
        """
        self._require_wallet()
        raw = self._wallet.balance(asset.lower())
        # CDP returns a Decimal or string — normalise to float
        return float(raw) if raw else 0.0

    def get_all_balances(self) -> dict[str, float]:
        """Return a dict of all non-zero asset balances."""
        self._require_wallet()
        raw = self._wallet.balances()
        return {asset: float(amount) for asset, amount in raw.items() if float(amount) > 0}

    def get_address(self) -> str:
        """Return the wallet's deposit address."""
        self._require_wallet()
        return self._address

    def get_wallet_info(self) -> dict:
        """Return a summary of wallet metadata."""
        self._require_wallet()
        return {
            "wallet_id": self._wallet_id,
            "address":   self._address,
            "network":   self._network,
            "balances":  self.get_all_balances(),
        }

    # ── Transactions ─────────────────────────────────────────────────────────

    def transfer(
        self,
        amount: float,
        asset: str,
        destination: str,
        portfolio_value_usd: float,
        reason: str = "agent_transfer",
    ) -> dict:
        """
        Transfer an asset to another address.

        The safety layer here is critical: before executing, we check that
        the USD value of the transfer doesn't exceed MAX_TRANSFER_PCT of
        the total portfolio. This prevents a misconfigured agent from
        draining the entire wallet in one call.

        Args:
            amount:              How many units of the asset to send.
            asset:               Asset ticker: 'eth', 'usdc', etc.
            destination:         The recipient wallet address (0x...).
            portfolio_value_usd: Total portfolio value (used for safety check).
            reason:              Human-readable label for the trade log.
        """
        self._require_wallet()

        # ── Safety check: estimate USD value and compare to limit ────────────
        # We use a rough price map; in production, call get_crypto_price() first.
        approx_prices_usd = {"eth": 3500.0, "usdc": 1.0, "cbeth": 3600.0}
        price_usd = approx_prices_usd.get(asset.lower(), 1.0)
        transfer_usd = amount * price_usd
        max_allowed_usd = portfolio_value_usd * (MAX_TRANSFER_PCT / 100)

        if transfer_usd > max_allowed_usd:
            return {
                "status":  "rejected",
                "reason":  f"Transfer value ${transfer_usd:,.2f} exceeds safety limit "
                           f"${max_allowed_usd:,.2f} ({MAX_TRANSFER_PCT}% of portfolio).",
                "tip":     "Increase CDP_MAX_TRANSFER_PCT in .env or reduce transfer amount.",
            }

        # ── Check we actually have the balance ───────────────────────────────
        available = self.get_balance(asset)
        if available < amount:
            return {
                "status": "rejected",
                "reason": f"Insufficient {asset.upper()} balance. "
                          f"Available: {available:.6f}, Requested: {amount:.6f}",
            }

        # ── Execute the transfer ─────────────────────────────────────────────
        try:
            transfer_obj = self._wallet.transfer(amount, asset, destination).wait()
            return {
                "status":         "success",
                "transfer_id":    transfer_obj.transfer_id,
                "transaction_hash": transfer_obj.transaction_hash,
                "transaction_link": transfer_obj.transaction_link,
                "amount":         amount,
                "asset":          asset.upper(),
                "destination":    destination,
                "approx_usd":     round(transfer_usd, 2),
                "reason":         reason,
                "timestamp":      datetime.utcnow().isoformat(),
                "network":        self._network,
            }
        except ApiError as e:
            return {"status": "error", "reason": str(e)}

    def request_faucet(self, asset: str = "eth") -> dict:
        """
        Request test tokens from the Coinbase faucet (testnet only).
        Use this to fund your Base Sepolia wallet for testing.
        """
        self._require_wallet()
        if "mainnet" in self._network:
            return {"status": "error", "reason": "Faucet only available on testnet networks."}
        try:
            tx = self._wallet.faucet(asset)
            return {
                "status": "success",
                "message": f"Faucet request sent for {asset.upper()} on {self._network}.",
                "transaction": str(tx),
            }
        except ApiError as e:
            return {"status": "error", "reason": str(e)}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _require_wallet(self):
        if not self.is_loaded():
            raise RuntimeError(
                "Wallet not loaded. Call manager.load() first, or "
                "manager.create_and_save() for first-time setup."
            )


# ─────────────────────────────────────────────────────────────────────────────
# LangChain @tool wrappers — these plug directly into your agent graph
# ─────────────────────────────────────────────────────────────────────────────
#
# The lazy_manager pattern is used here because we don't want the wallet
# to load at import time (which would fail if CDP isn't configured yet).
# Instead, the wallet loads on the first tool call, then stays cached.

_manager: Optional[WalletManager] = None

def _get_manager() -> WalletManager:
    """Return the cached WalletManager, loading it on first access."""
    global _manager
    if _manager is None:
        _manager = WalletManager()
        _manager.load()
    return _manager


@tool
def cdp_get_balance(asset: str = "eth") -> str:
    """
    Check the current balance of a specific asset in the CDP trading wallet.

    Use this before the Risk Manager calculates position size — knowing
    the actual available balance prevents recommending a trade the wallet
    can't fund.

    Args:
        asset: Asset ticker, e.g. 'eth', 'usdc', 'cbeth'. Default is 'eth'.

    Returns a plain string describing the balance, suitable for the agent.
    """
    try:
        mgr     = _get_manager()
        balance = mgr.get_balance(asset)
        address = mgr.get_address()
        return (
            f"CDP Wallet Balance\n"
            f"  Asset:   {asset.upper()}\n"
            f"  Amount:  {balance:.6f}\n"
            f"  Address: {address}\n"
            f"  Network: {mgr._network}"
        )
    except Exception as e:
        return f"Error fetching balance: {e}"


@tool
def cdp_get_all_balances() -> str:
    """
    Retrieve all non-zero asset balances in the CDP trading wallet.

    Use this at the start of any session to get a complete picture of
    what's available before making any recommendations.
    """
    try:
        mgr      = _get_manager()
        balances = mgr.get_all_balances()
        if not balances:
            return "CDP wallet has no funded balances. Use cdp_request_faucet() on testnet."
        lines = ["CDP Wallet — All Balances:"]
        for asset, amount in balances.items():
            lines.append(f"  {asset.upper():<8} {amount:.6f}")
        lines.append(f"\nAddress: {mgr.get_address()}")
        lines.append(f"Network: {mgr._network}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching balances: {e}"


@tool
def cdp_transfer(
    amount: float,
    asset: str,
    destination: str,
    portfolio_value_usd: float,
    reason: str = "agent_trade",
) -> str:
    """
    Transfer an asset from the CDP wallet to another address.

    This tool has a built-in safety check: the transfer will be rejected
    if its estimated USD value exceeds the configured maximum percentage
    of the total portfolio. This prevents runaway agent behaviour.

    Args:
        amount:              Number of asset units to transfer (e.g. 0.01 for 0.01 ETH).
        asset:               Asset ticker: 'eth', 'usdc', etc.
        destination:         Recipient wallet address (0x...).
        portfolio_value_usd: Your total portfolio value in USD (for safety check).
        reason:              Label for the transaction log.

    Returns a JSON-formatted result string with status, tx hash, and details.
    """
    try:
        mgr    = _get_manager()
        result = mgr.transfer(amount, asset, destination, portfolio_value_usd, reason)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "reason": str(e)})


@tool
def cdp_get_wallet_info() -> str:
    """
    Return full metadata about the CDP trading wallet: ID, address,
    network, and all current balances.

    Use this when the user asks about their wallet or to verify the
    system is connected to the correct on-chain wallet.
    """
    try:
        mgr  = _get_manager()
        info = mgr.get_wallet_info()
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error fetching wallet info: {e}"


@tool
def cdp_request_faucet(asset: str = "eth") -> str:
    """
    Request free test tokens from the Coinbase faucet (testnet only).

    Use this to fund your Base Sepolia wallet for paper trading and
    testing without risking real money. Will fail on mainnet.

    Args:
        asset: Asset to request. 'eth' is the most useful on Base Sepolia.
    """
    try:
        mgr    = _get_manager()
        result = mgr.request_faucet(asset)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "reason": str(e)})


# Convenience export: add these to any agent's tool list
CDP_TOOLS = [
    cdp_get_balance,
    cdp_get_all_balances,
    cdp_get_wallet_info,
    cdp_request_faucet,
    cdp_transfer,  # ⚠️ only add to agents that should execute trades
]

# Read-only tools — safe for Market Analyst and Risk Manager
CDP_READONLY_TOOLS = [
    cdp_get_balance,
    cdp_get_all_balances,
    cdp_get_wallet_info,
    cdp_request_faucet,
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper — run once to initialise your wallet
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="CDP Wallet Manager CLI")
    parser.add_argument("--init",     action="store_true", help="Create a new wallet")
    parser.add_argument("--info",     action="store_true", help="Show wallet info")
    parser.add_argument("--balance",  type=str, default=None, help="Check balance of asset")
    parser.add_argument("--faucet",   type=str, default=None, help="Request faucet for asset")
    parser.add_argument("--network",  type=str, default=NETWORK_ID, help="Network ID")
    args = parser.parse_args()

    if args.init:
        mgr = WalletManager()
        mgr.create_and_save(network_id=args.network)

    elif args.info:
        mgr = WalletManager()
        mgr.load()
        print(_json.dumps(mgr.get_wallet_info(), indent=2))

    elif args.balance:
        mgr = WalletManager()
        mgr.load()
        bal = mgr.get_balance(args.balance)
        print(f"{args.balance.upper()} balance: {bal}")

    elif args.faucet:
        mgr = WalletManager()
        mgr.load()
        result = mgr.request_faucet(args.faucet)
        print(_json.dumps(result, indent=2))

    else:
        parser.print_help()
