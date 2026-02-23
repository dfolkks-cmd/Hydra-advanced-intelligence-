"""
tools/crypto_tools.py — LangChain tools available to specialist agents.

Each tool is decorated with @tool so LangGraph can bind them to agents
via bind_tools(). Tools cover: price data, news, technical indicators,
portfolio maths, and a sandboxed Python executor.
"""

from __future__ import annotations

import io
import json
import math
import sys
import traceback
from contextlib import redirect_stdout
from typing import Optional

import numpy as np
import pandas as pd
import requests
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_crypto_price(symbol: str) -> str:
    """
    Fetch the current price, 24h change, market cap, and volume for a
    cryptocurrency symbol (e.g. 'BTC', 'ETH', 'SOL').

    Returns a JSON string with price data from CoinGecko's free API.
    """
    coin_map = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot",
        "LINK": "chainlink", "MATIC": "matic-network", "UNI": "uniswap",
        "LTC": "litecoin", "ATOM": "cosmos", "NEAR": "near",
        "ARB": "arbitrum", "OP": "optimism", "SUI": "sui",
        "INJ": "injective-protocol", "TIA": "celestia",
    }
    coin_id = coin_map.get(symbol.upper(), symbol.lower())
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            "?localization=false&tickers=false&community_data=false"
            "&developer_data=false&sparkline=false"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        md = d["market_data"]
        return json.dumps({
            "symbol": symbol.upper(),
            "name": d["name"],
            "price_usd": md["current_price"]["usd"],
            "price_change_24h_pct": md["price_change_percentage_24h"],
            "price_change_7d_pct": md["price_change_percentage_7d"],
            "price_change_30d_pct": md["price_change_percentage_30d"],
            "market_cap_usd": md["market_cap"]["usd"],
            "volume_24h_usd": md["total_volume"]["usd"],
            "ath": md["ath"]["usd"],
            "ath_change_pct": md["ath_change_percentage"]["usd"],
            "circulating_supply": md["circulating_supply"],
            "market_cap_rank": d["market_cap_rank"],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


@tool
def get_fear_greed_index() -> str:
    """
    Fetch the current Crypto Fear & Greed Index (0 = extreme fear, 100 = extreme greed).
    Useful for gauging overall market sentiment.
    """
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=3", timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        return json.dumps({
            "current": {
                "value": int(data[0]["value"]),
                "classification": data[0]["value_classification"],
                "timestamp": data[0]["timestamp"],
            },
            "yesterday": {
                "value": int(data[1]["value"]),
                "classification": data[1]["value_classification"],
            },
            "last_week": {
                "value": int(data[2]["value"]),
                "classification": data[2]["value_classification"],
            },
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_historical_prices(symbol: str, days: int = 30) -> str:
    """
    Fetch daily OHLCV (Open/High/Low/Close/Volume) price history for a coin.
    Use this before running technical analysis.

    Args:
        symbol: Coin ticker (e.g. 'BTC', 'ETH')
        days:   Number of days of history (7, 14, 30, 90, 180, 365)
    """
    coin_map = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot",
        "LINK": "chainlink", "MATIC": "matic-network", "UNI": "uniswap",
        "LTC": "litecoin", "ATOM": "cosmos", "NEAR": "near",
        "ARB": "arbitrum", "OP": "optimism", "SUI": "sui",
    }
    coin_id = coin_map.get(symbol.upper(), symbol.lower())
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            f"?vs_currency=usd&days={days}"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()  # [[timestamp, open, high, low, close], ...]
        records = [
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
            }
            for row in raw
        ]
        return json.dumps({
            "symbol": symbol.upper(),
            "days": days,
            "records": records[-60:],   # cap at 60 candles to stay in context
        })
    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculate_technical_indicators(price_data_json: str) -> str:
    """
    Compute RSI, MACD, Bollinger Bands, and EMA (20/50) from OHLCV data.

    Args:
        price_data_json: JSON string returned by get_historical_prices()

    Returns a JSON string with all indicator values and simple signals.
    """
    try:
        data = json.loads(price_data_json)
        if "error" in data:
            return json.dumps({"error": data["error"]})

        records = data["records"]
        if len(records) < 26:
            return json.dumps({"error": "Not enough data (need ≥26 candles)"})

        closes = pd.Series([r["close"] for r in records], dtype=float)
        highs  = pd.Series([r["high"]  for r in records], dtype=float)
        lows   = pd.Series([r["low"]   for r in records], dtype=float)

        # ── RSI ──────────────────────────────────────────────
        delta = closes.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]

        # ── MACD ─────────────────────────────────────────────
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram   = macd_line - signal_line
        macd_val    = macd_line.iloc[-1]
        sig_val     = signal_line.iloc[-1]
        hist_val    = histogram.iloc[-1]
        macd_signal = "buy" if macd_val > sig_val else "sell"

        # ── Bollinger Bands ───────────────────────────────────
        bb_mid   = closes.rolling(20).mean()
        bb_std   = closes.rolling(20).std()
        bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
        bb_lower = (bb_mid - 2 * bb_std).iloc[-1]
        bb_mid_v = bb_mid.iloc[-1]
        cur_price = closes.iloc[-1]

        if   cur_price > bb_upper:           bb_pos = "above_upper"
        elif cur_price > bb_mid_v * 1.005:   bb_pos = "near_upper"
        elif cur_price < bb_lower:           bb_pos = "below_lower"
        elif cur_price < bb_mid_v * 0.995:   bb_pos = "near_lower"
        else:                                bb_pos = "middle"

        # ── EMAs ─────────────────────────────────────────────
        ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1] if len(closes) >= 50 else None

        if ema50:
            if cur_price > ema20 > ema50:  ema_trend = "uptrend"
            elif cur_price < ema20 < ema50: ema_trend = "downtrend"
            else:                           ema_trend = "sideways"
        else:
            ema_trend = "uptrend" if cur_price > ema20 else "downtrend"

        # ── Support / Resistance (simple local min/max) ───────
        window = min(10, len(closes) // 4)
        local_min = closes.rolling(window, center=True).min()
        local_max = closes.rolling(window, center=True).max()
        supports   = sorted(set(round(v, 2) for v in local_min.dropna().tail(5)))
        resistances= sorted(set(round(v, 2) for v in local_max.dropna().tail(5)), reverse=True)

        return json.dumps({
            "symbol": data["symbol"],
            "current_price": round(cur_price, 4),
            "rsi": round(float(rsi), 2),
            "rsi_signal": "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral"),
            "macd": round(float(macd_val), 6),
            "macd_signal_line": round(float(sig_val), 6),
            "macd_histogram": round(float(hist_val), 6),
            "macd_signal": macd_signal,
            "bb_upper": round(float(bb_upper), 4),
            "bb_middle": round(float(bb_mid_v), 4),
            "bb_lower": round(float(bb_lower), 4),
            "bb_position": bb_pos,
            "ema_20": round(float(ema20), 4),
            "ema_50": round(float(ema50), 4) if ema50 else None,
            "ema_trend": ema_trend,
            "support_levels": supports[:3],
            "resistance_levels": resistances[:3],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculate_position_size(
    portfolio_value: float,
    risk_tolerance: str,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_pct: Optional[float] = None,
) -> str:
    """
    Calculate the optimal position size using the Kelly Criterion variant
    and fixed-fractional risk management.

    Args:
        portfolio_value:    Total portfolio in USD
        risk_tolerance:     'conservative', 'moderate', or 'aggressive'
        entry_price:        Planned entry price
        stop_loss_price:    Planned stop-loss price
        risk_per_trade_pct: Override the default risk % per trade (0–5)
    """
    risk_map = {"conservative": 1.0, "moderate": 2.0, "aggressive": 3.0}
    default_risk = risk_map.get(risk_tolerance.lower(), 2.0)
    risk_pct = risk_per_trade_pct if risk_per_trade_pct else default_risk

    risk_amount = portfolio_value * (risk_pct / 100)
    price_risk  = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        return json.dumps({"error": "Entry and stop-loss prices cannot be equal"})

    units = risk_amount / price_risk
    position_usd = units * entry_price
    max_position_pct = {"conservative": 10, "moderate": 20, "aggressive": 30}.get(
        risk_tolerance.lower(), 20
    )
    max_position_usd = portfolio_value * max_position_pct / 100

    if position_usd > max_position_usd:
        position_usd = max_position_usd
        units = position_usd / entry_price

    stop_pct   = abs(entry_price - stop_loss_price) / entry_price * 100
    target_pct = stop_pct * {"conservative": 2, "moderate": 3, "aggressive": 4}.get(
        risk_tolerance.lower(), 2
    )
    target_price = entry_price * (1 + target_pct / 100)
    rr_ratio     = target_pct / stop_pct

    return json.dumps({
        "portfolio_value_usd": portfolio_value,
        "risk_per_trade_pct": risk_pct,
        "risk_amount_usd": round(risk_amount, 2),
        "units_to_buy": round(units, 6),
        "position_size_usd": round(position_usd, 2),
        "position_pct_of_portfolio": round(position_usd / portfolio_value * 100, 2),
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "stop_loss_pct_from_entry": round(stop_pct, 2),
        "take_profit_price": round(target_price, 4),
        "take_profit_pct_from_entry": round(target_pct, 2),
        "risk_reward_ratio": round(rr_ratio, 2),
    }, indent=2)


@tool
def assess_portfolio_risk(holdings_json: str, portfolio_value: float) -> str:
    """
    Assess concentration risk, correlation risk, and overall portfolio health.

    Args:
        holdings_json:   JSON string like '{"BTC": 0.5, "ETH": 2.0, "SOL": 10}'
                         where values are coin quantities
        portfolio_value: Total portfolio value in USD
    """
    try:
        holdings = json.loads(holdings_json)
        warnings = []

        # Approximate USD values using rough price map (real app would call API)
        approx_prices = {
            "BTC": 95000, "ETH": 3500, "SOL": 200, "BNB": 600,
            "XRP": 0.6, "ADA": 0.4, "DOGE": 0.15, "AVAX": 40,
            "LINK": 15, "DOT": 7, "MATIC": 0.5, "UNI": 10,
        }

        allocations = {}
        for coin, qty in holdings.items():
            price = approx_prices.get(coin.upper(), 1.0)
            allocations[coin.upper()] = qty * price

        total_known = sum(allocations.values())
        unknown_value = max(0, portfolio_value - total_known)
        if unknown_value > 0:
            allocations["OTHER"] = unknown_value

        total = sum(allocations.values())
        pcts  = {k: round(v / total * 100, 2) for k, v in allocations.items()}

        # Concentration check
        for coin, pct in pcts.items():
            if pct > 40:
                warnings.append(f"HIGH CONCENTRATION: {coin} is {pct:.1f}% of portfolio")
            elif pct > 25:
                warnings.append(f"Moderate concentration: {coin} at {pct:.1f}%")

        # Alt-heavy check
        btc_pct = pcts.get("BTC", 0)
        eth_pct = pcts.get("ETH", 0)
        alt_pct = 100 - btc_pct - eth_pct - pcts.get("OTHER", 0)
        if alt_pct > 60:
            warnings.append(f"High altcoin exposure ({alt_pct:.1f}%) — elevated risk in bear markets")

        # Determine risk level
        max_single = max(pcts.values())
        if   max_single > 60: risk_level = "extreme"
        elif max_single > 40: risk_level = "high"
        elif max_single > 25: risk_level = "medium"
        else:                 risk_level = "low"

        return json.dumps({
            "portfolio_value_usd": portfolio_value,
            "allocations_pct": pcts,
            "risk_level": risk_level,
            "btc_dominance_in_portfolio": btc_pct,
            "eth_exposure_pct": eth_pct,
            "altcoin_exposure_pct": round(alt_pct, 2),
            "warnings": warnings,
            "diversification_score": round(10 - (max_single / 10), 1),  # 0–10
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# CODE EXECUTION TOOL
# ─────────────────────────────────────────────────────────────────────────────

@tool
def execute_python(code: str) -> str:
    """
    Execute Python code in a sandboxed environment with access to:
    numpy, pandas, math. Use for custom calculations, backtesting
    snippets, or data transformations.

    Args:
        code: Python source code as a string. Print results to stdout.

    Returns stdout output (capped at 4000 chars).
    """
    allowed_globals = {
        "__builtins__": {
            k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
            for k in [
                "print", "range", "len", "sum", "min", "max", "abs", "round",
                "int", "float", "str", "list", "dict", "tuple", "set", "bool",
                "zip", "enumerate", "sorted", "reversed", "isinstance",
                "type", "repr", "format", "map", "filter",
            ]
        },
        "np": np,
        "pd": pd,
        "math": math,
        "json": json,
    }
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, allowed_globals)   # noqa: S102
        output = buf.getvalue()
    except Exception as e:
        output = f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    return output[:4000]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

# Convenient grouped exports for bind_tools()
MARKET_TOOLS     = [get_crypto_price, get_fear_greed_index, get_historical_prices]
TECHNICAL_TOOLS  = [get_historical_prices, calculate_technical_indicators, execute_python]
RISK_TOOLS       = [calculate_position_size, assess_portfolio_risk]
STRATEGY_TOOLS   = [get_crypto_price, get_fear_greed_index, execute_python]
ALL_TOOLS        = [
    get_crypto_price,
    get_fear_greed_index,
    get_historical_prices,
    calculate_technical_indicators,
    calculate_position_size,
    assess_portfolio_risk,
    execute_python,
]
