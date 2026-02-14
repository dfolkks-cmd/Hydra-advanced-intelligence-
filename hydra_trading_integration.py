#!/usr/bin/env python3
"""
HYDRA TRADING INTEGRATION
Connects Cognitive Agents to Market Analysis & Risk Management
"""

import asyncio
import random
from typing import Dict, List
from colorama import Fore, Style
from hydra_cognitive_system import CognitiveAgent, EnhancedMemory


class MarketAnalyzer:
    """Simulates market data analysis (replace with real exchange API)"""
    
    def __init__(self):
        self.price_history = []
    
    def analyze_market(self, symbol: str) -> Dict:
        """
        Analyzes current market conditions
        In production: Connect to exchange APIs (Coinbase, Binance, etc.)
        """
        # Simulate market data
        price = 50000 + random.uniform(-5000, 5000)
        volume = random.uniform(1000000, 5000000)
        
        # Technical indicators (simplified)
        rsi = random.uniform(20, 80)
        macd = random.uniform(-100, 100)
        
        # Pattern detection
        patterns = []
        if rsi > 70:
            patterns.append("overbought")
        elif rsi < 30:
            patterns.append("oversold")
        
        if macd > 50:
            patterns.append("bullish_momentum")
        elif macd < -50:
            patterns.append("bearish_momentum")
        
        # Volume analysis
        if volume > 3000000:
            patterns.append("high_volume")
        
        return {
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "rsi": rsi,
            "macd": macd,
            "patterns": patterns,
            "trend": "bullish" if macd > 0 else "bearish"
        }


class RiskManager:
    """Manages position sizing and risk limits"""
    
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.05):
        self.max_position_size = max_position_size  # 10% of portfolio
        self.max_drawdown = max_drawdown  # 5% max loss
        self.portfolio_value = 10000  # Starting capital
        self.positions = {}
    
    def calculate_position_size(self, confidence: float, volatility: float) -> float:
        """
        Kelly Criterion-inspired position sizing
        """
        base_size = self.portfolio_value * self.max_position_size
        
        # Adjust for confidence
        confidence_factor = min(confidence, 0.8)  # Cap at 80%
        
        # Adjust for volatility
        volatility_factor = max(1.0 - volatility, 0.2)
        
        position_size = base_size * confidence_factor * volatility_factor
        return position_size
    
    def check_drawdown(self) -> bool:
        """Returns True if within acceptable drawdown"""
        current_value = self.portfolio_value + sum(self.positions.values())
        drawdown = (10000 - current_value) / 10000
        return drawdown < self.max_drawdown


class HydraCouncil:
    """
    Orchestrates multiple agents for collective decision making
    """
    
    def __init__(self):
        self.memory = EnhancedMemory("hydra_trading_memory.json")
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        
        # Initialize agent council with diverse personalities
        self.agents = {
            "STINKMEANER": CognitiveAgent(
                "STINKMEANER", 
                "Aggressor",
                {"aggression": 0.9, "patience": 0.2, "greed": 0.8, "fear": 0.2},
                Fore.RED
            ),
            "SAMUEL": CognitiveAgent(
                "SAMUEL",
                "Strategist", 
                {"aggression": 0.5, "patience": 0.8, "greed": 0.4, "fear": 0.4},
                Fore.YELLOW
            ),
            "CLAYTON": CognitiveAgent(
                "CLAYTON",
                "Conservator",
                {"aggression": 0.2, "patience": 0.9, "greed": 0.3, "fear": 0.7},
                Fore.WHITE
            )
        }
    
    async def analyze_opportunity(self, symbol: str) -> Dict:
        """
        Full analysis pipeline:
        1. Get market data
        2. Each agent analyzes
        3. Consensus vote
        4. Risk management
        """
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}🎯 ANALYZING: {symbol}")
        print(f"{Fore.CYAN}{'='*70}\n")
        
        # Step 1: Market Analysis
        market_data = self.market_analyzer.analyze_market(symbol)
        
        print(f"{Fore.WHITE}📊 Market Data:")
        print(f"   Price: ${market_data['price']:,.2f}")
        print(f"   Volume: ${market_data['volume']:,.0f}")
        print(f"   RSI: {market_data['rsi']:.1f}")
        print(f"   MACD: {market_data['macd']:.1f}")
        print(f"   Patterns: {', '.join(market_data['patterns']) if market_data['patterns'] else 'None'}")
        print(f"   Trend: {market_data['trend'].upper()}\n")
        
        # Step 2: Generate context for agents
        context = self._generate_context(market_data)
        
        # Step 3: Each agent votes
        votes = {}
        confidences = []
        
        for name, agent in self.agents.items():
            result = agent.think(context, self.memory)
            votes[name] = result['decision']
            confidences.append(result['confidence'])
            
            agent.speak(f"Vote: {result['decision']} (Confidence: {result['confidence']:.0%})")
            print(f"{Fore.BLUE}   Reasoning: {result['final_thought'][:100]}...\n")
        
        # Step 4: Consensus calculation
        decision, consensus_strength = self._calculate_consensus(votes)
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"\n{Fore.GREEN}{'─'*70}")
        print(f"{Fore.GREEN}🏛️  COUNCIL DECISION: {decision}")
        print(f"{Fore.GREEN}   Consensus Strength: {consensus_strength:.0%}")
        print(f"{Fore.GREEN}   Average Confidence: {avg_confidence:.0%}")
        print(f"{Fore.GREEN}{'─'*70}\n")
        
        # Step 5: Risk Management
        if decision in ["BUY", "SMALL_POSITION"]:
            position_size = self.risk_manager.calculate_position_size(
                avg_confidence, 
                self._estimate_volatility(market_data)
            )
            
            print(f"{Fore.YELLOW}💰 Risk Management:")
            print(f"   Position Size: ${position_size:,.2f}")
            print(f"   % of Portfolio: {(position_size/self.risk_manager.portfolio_value)*100:.1f}%\n")
        
        return {
            "symbol": symbol,
            "decision": decision,
            "consensus": consensus_strength,
            "confidence": avg_confidence,
            "market_data": market_data
        }
    
    def _generate_context(self, market_data: Dict) -> str:
        """Convert market data to natural language context"""
        patterns = market_data['patterns']
        context_parts = [f"{market_data['symbol']} is {market_data['trend']}"]
        
        if "high_volume" in patterns:
            context_parts.append("with unusually high volume")
        
        if "overbought" in patterns:
            context_parts.append("showing overbought conditions (RSI > 70)")
        elif "oversold" in patterns:
            context_parts.append("showing oversold conditions (RSI < 30)")
        
        if "bullish_momentum" in patterns:
            context_parts.append("with strong bullish momentum")
        elif "bearish_momentum" in patterns:
            context_parts.append("with bearish momentum building")
        
        return " ".join(context_parts)
    
    def _calculate_consensus(self, votes: Dict[str, str]) -> tuple:
        """
        Calculate consensus from agent votes
        Returns: (decision, strength)
        """
        vote_counts = {}
        for vote in votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        max_votes = max(vote_counts.values())
        total_votes = len(votes)
        consensus_strength = max_votes / total_votes
        
        # Get majority decision
        for decision, count in vote_counts.items():
            if count == max_votes:
                return decision, consensus_strength
        
        return "HOLD", 0.33
    
    def _estimate_volatility(self, market_data: Dict) -> float:
        """Estimate volatility from market data"""
        # Simple volatility estimate based on RSI extremes
        rsi = market_data['rsi']
        if rsi > 70 or rsi < 30:
            return 0.8  # High volatility
        elif 40 <= rsi <= 60:
            return 0.3  # Low volatility
        else:
            return 0.5  # Medium volatility
    
    async def run_trading_session(self, symbols: List[str], cycles: int = 3):
        """
        Run a full trading session analyzing multiple symbols
        """
        print(f"\n{Fore.MAGENTA}{'='*70}")
        print(f"{Fore.MAGENTA}🚀 HYDRA TRADING SESSION INITIATED")
        print(f"{Fore.MAGENTA}   Analyzing {len(symbols)} symbols over {cycles} cycles")
        print(f"{Fore.MAGENTA}{'='*70}\n")
        
        for cycle in range(cycles):
            print(f"\n{Fore.CYAN}{Style.BRIGHT}═══ CYCLE {cycle + 1}/{cycles} ═══{Style.RESET_ALL}\n")
            
            for symbol in symbols:
                result = await self.analyze_opportunity(symbol)
                
                # Record significant decisions in memory
                if result['consensus'] > 0.66 and result['confidence'] > 0.6:
                    impact = 0.7 if result['decision'] == "BUY" else 0.3
                    self.memory.add_episode(
                        "COUNCIL",
                        f"Strong consensus to {result['decision']} {symbol}",
                        impact,
                        result['market_data']['patterns'] + [symbol.lower()],
                        result['market_data']
                    )
                
                await asyncio.sleep(1)  # Rate limiting
            
            print(f"\n{Fore.BLUE}{'─'*70}")
            print(f"{Fore.BLUE}⏸️  Cycle {cycle + 1} complete. Cooling down...")
            print(f"{Fore.BLUE}{'─'*70}\n")
            await asyncio.sleep(2)
        
        # Final statistics
        stats = self.memory.get_statistics()
        print(f"\n{Fore.GREEN}{'='*70}")
        print(f"{Fore.GREEN}📊 SESSION COMPLETE")
        print(f"{Fore.GREEN}{'='*70}")
        print(f"{Fore.WHITE}Total Decisions Made: {stats.get('total_episodes', 0)}")
        print(f"Portfolio Value: ${self.risk_manager.portfolio_value:,.2f}")
        print(f"Win/Loss Record: {stats.get('total_wins', 0)}/{stats.get('total_losses', 0)}")
        print(f"{Fore.GREEN}{'='*70}\n")


async def main():
    """Main entry point"""
    council = HydraCouncil()
    
    # Symbols to analyze
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    await council.run_trading_session(symbols, cycles=2)


if __name__ == "__main__":
    asyncio.run(main())
