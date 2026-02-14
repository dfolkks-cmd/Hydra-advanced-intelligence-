#!/usr/bin/env python3
"""
HYDRA COGNITIVE SYSTEM V2.0
Enhanced Agent Architecture with Episodic Memory & Chain-of-Thought Reasoning
"""

import time
import json
import math
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init

init(autoreset=True)

# ==============================================================================
# ENHANCED MEMORY: EPISODIC + SEMANTIC LAYERS
# ==============================================================================

class EnhancedMemory:
    """
    Implements a two-tiered memory system:
    1. Short-term (Working Memory): Context window for immediate reasoning.
    2. Long-term (Episodic): Significant events (Wins/Losses) with 'Emotional Weight'.
    """
    def __init__(self, filepath="hydra_enhanced_memory.json"):
        self.filepath = filepath
        self.working_memory = []
        self.episodic_memory = self._load()
        
    def _load(self) -> Dict:
        """Load memory from disk or create new"""
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "episodes": [], 
                "knowledge_graph": {},
                "metadata": {
                    "total_wins": 0,
                    "total_losses": 0,
                    "created": time.time()
                }
            }

    def save(self):
        """Persist memory to disk"""
        with open(self.filepath, 'w') as f:
            json.dump(self.episodic_memory, f, indent=2)

    def add_episode(self, agent: str, event: str, emotional_impact: float, 
                    tags: List[str], metadata: Optional[Dict] = None):
        """
        Add a new episodic memory
        
        Args:
            agent: Name of the agent
            event: Description of what happened
            emotional_impact: -1.0 (Trauma) to 1.0 (Euphoria)
            tags: Keywords for retrieval (e.g., "crash", "gas_fees", "breakout")
            metadata: Optional additional data (profit/loss, price, etc.)
        """
        episode = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "agent": agent,
            "event": event,
            "impact": emotional_impact,
            "tags": tags,
            "metadata": metadata or {}
        }
        self.episodic_memory["episodes"].append(episode)
        
        # Update stats
        if emotional_impact > 0:
            self.episodic_memory["metadata"]["total_wins"] += 1
        elif emotional_impact < 0:
            self.episodic_memory["metadata"]["total_losses"] += 1
        
        # Prune memory if too large (keep most impactful events)
        if len(self.episodic_memory["episodes"]) > 1000:
            self.episodic_memory["episodes"].sort(
                key=lambda x: abs(x["impact"]), 
                reverse=True
            )
            self.episodic_memory["episodes"] = self.episodic_memory["episodes"][:800]
        
        self.save()

    def retrieve_relevant(self, context_tags: List[str], limit: int = 3) -> List[Dict]:
        """
        Retrieves memories that match the current context tags.
        Simulates 'Associative Memory' with scoring algorithm.
        """
        scored_memories = []
        current_time = time.time()
        
        for episode in self.episodic_memory["episodes"]:
            score = 0.0
            
            # 1. Keyword matching (semantic similarity)
            tag_overlap = len(set(episode["tags"]) & set(context_tags))
            score += tag_overlap * 2.0
            
            # 2. Recency bias (exponential decay)
            age_hours = (current_time - episode["timestamp"]) / 3600
            recency_score = 1.0 / (1.0 + age_hours * 0.1)
            score += recency_score
            
            # 3. Emotional intensity (vivid memories stick)
            intensity_score = abs(episode["impact"]) * 1.5
            score += intensity_score
            
            if score > 0.5:
                scored_memories.append((score, episode))
        
        # Sort by relevance score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in scored_memories[:limit]]

    def get_statistics(self) -> Dict:
        """Return memory statistics"""
        total = len(self.episodic_memory["episodes"])
        if total == 0:
            return {"total": 0, "avg_impact": 0, "wins": 0, "losses": 0}
        
        impacts = [e["impact"] for e in self.episodic_memory["episodes"]]
        return {
            "total_episodes": total,
            "avg_impact": sum(impacts) / total,
            "total_wins": self.episodic_memory["metadata"]["total_wins"],
            "total_losses": self.episodic_memory["metadata"]["total_losses"],
            "most_recent": self.episodic_memory["episodes"][-1] if total > 0 else None
        }


# ==============================================================================
# COGNITIVE AGENT: "SYSTEM 2" REASONING ENGINE
# ==============================================================================

class CognitiveAgent:
    """
    Advanced agent with dual-process thinking:
    - System 1: Fast, intuitive reactions
    - System 2: Slow, deliberate reasoning with memory integration
    """
    
    def __init__(self, name: str, role: str, personality_traits: Dict[float, float], 
                 color: str):
        self.name = name
        self.role = role
        self.traits = personality_traits  # {"aggression": 0.8, "patience": 0.2, ...}
        self.color = color
        self.decision_history = []
        
    def think(self, context: str, memory_system: EnhancedMemory) -> Dict[str, Any]:
        """
        The 'Three-Step' Rethink Process (Chain-of-Thought)
        
        Returns:
            Dict with reasoning chain and final decision
        """
        # Step 1: Initial Reaction (System 1 - Fast)
        reaction = self._generate_reaction(context)
        
        # Step 2: Context Retrieval (Memory Lookup)
        tags = self._extract_tags(context)
        memories = memory_system.retrieve_relevant(tags)
        
        # Step 3: Synthesis & Refinement (System 2 - Slow)
        final_thought, decision = self._synthesize(reaction, memories, context)
        
        # Record decision
        decision_record = {
            "timestamp": time.time(),
            "context": context,
            "reaction": reaction,
            "memories_used": len(memories),
            "final_thought": final_thought,
            "decision": decision
        }
        self.decision_history.append(decision_record)
        
        return {
            "reaction": reaction,
            "memories": memories,
            "final_thought": final_thought,
            "decision": decision,
            "confidence": self._calculate_confidence(memories)
        }

    def _extract_tags(self, context: str) -> List[str]:
        """Extract relevant keywords from context"""
        # Simple keyword extraction (can be enhanced with NLP)
        keywords = context.lower().split()
        
        # Trading-specific patterns
        patterns = {
            "breakout", "breakdown", "support", "resistance", 
            "pump", "dump", "crash", "moon", "dip", "rally",
            "volume", "momentum", "reversal", "trend"
        }
        
        return [w for w in keywords if w in patterns or len(w) > 4]

    def _generate_reaction(self, context: str) -> str:
        """System 1: Fast, instinctive response based on personality"""
        aggression = self.traits.get("aggression", 0.5)
        patience = self.traits.get("patience", 0.5)
        greed = self.traits.get("greed", 0.5)
        fear = self.traits.get("fear", 0.5)
        
        context_lower = context.lower()
        
        # Pattern matching for quick reactions
        if "breakout" in context_lower or "pump" in context_lower:
            if aggression > 0.7:
                return "IMMEDIATE BUY - This is our moment!"
            elif patience > 0.7:
                return "Interesting pattern. Let's confirm first."
            else:
                return "Possible opportunity detected."
                
        elif "crash" in context_lower or "dump" in context_lower:
            if fear > 0.7:
                return "DANGER! Exit everything NOW!"
            elif greed > 0.7:
                return "Blood in the streets = BUY TIME!"
            else:
                return "Market volatility detected."
        
        return f"Analyzing: {context}"

    def _synthesize(self, reaction: str, memories: List[Dict], 
                    context: str) -> tuple[str, str]:
        """
        System 2: Deliberate reasoning that integrates memories
        
        Returns:
            (final_thought, decision) tuple
        """
        if not memories:
            return reaction, self._reaction_to_decision(reaction)
        
        # Analyze memory patterns
        trauma_memories = [m for m in memories if m["impact"] < -0.5]
        winning_memories = [m for m in memories if m["impact"] > 0.5]
        
        # Check for pattern matching with past trauma
        if trauma_memories:
            most_traumatic = max(trauma_memories, key=lambda x: abs(x["impact"]))
            
            # Override reaction if similar pattern
            if self._is_similar_context(context, most_traumatic["event"]):
                final = (
                    f"{reaction} ...WAIT. This reminds me of {most_traumatic['event']}. "
                    f"We lost badly (impact: {most_traumatic['impact']:.2f}). "
                    f"I'm overriding my initial instinct."
                )
                return final, "HOLD"
        
        # Check for pattern matching with past wins
        if winning_memories:
            best_win = max(winning_memories, key=lambda x: x["impact"])
            
            if self._is_similar_context(context, best_win["event"]):
                final = (
                    f"{reaction} AND THIS IS JUST LIKE {best_win['event']}! "
                    f"That trade made us {best_win['impact']:.2f}. "
                    f"Pattern recognition says GO!"
                )
                return final, "BUY"
        
        # Default: Modified reaction based on memory sentiment
        avg_impact = sum(m["impact"] for m in memories) / len(memories)
        
        if avg_impact < -0.3:
            final = f"{reaction} ...but historical data suggests caution. Risk: HIGH"
            decision = "WAIT"
        elif avg_impact > 0.3:
            final = f"{reaction} ...and past performance supports this move!"
            decision = "BUY"
        else:
            final = f"{reaction} ...mixed signals from memory. Proceed with caution."
            decision = "SMALL_POSITION"
        
        return final, decision

    def _is_similar_context(self, current: str, past: str) -> bool:
        """Simple similarity check (can be enhanced)"""
        current_words = set(current.lower().split())
        past_words = set(past.lower().split())
        overlap = len(current_words & past_words)
        return overlap >= 2

    def _reaction_to_decision(self, reaction: str) -> str:
        """Convert reaction to actionable decision"""
        reaction_lower = reaction.lower()
        if "buy" in reaction_lower or "moment" in reaction_lower:
            return "BUY"
        elif "sell" in reaction_lower or "exit" in reaction_lower:
            return "SELL"
        elif "wait" in reaction_lower or "confirm" in reaction_lower:
            return "WAIT"
        else:
            return "HOLD"

    def _calculate_confidence(self, memories: List[Dict]) -> float:
        """Calculate decision confidence based on memory consistency"""
        if not memories:
            return 0.5
        
        impacts = [m["impact"] for m in memories]
        avg = sum(impacts) / len(impacts)
        variance = sum((x - avg) ** 2 for x in impacts) / len(impacts)
        
        # Low variance = high confidence
        confidence = 1.0 / (1.0 + variance)
        return min(max(confidence, 0.1), 0.95)

    def speak(self, text: str):
        """Output agent's thoughts with personality styling"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Fore.WHITE}[{timestamp}] {self.color}{Style.BRIGHT}[{self.name}]: {text}{Style.RESET_ALL}")
        time.sleep(0.3)  # Dramatic pause


# ==============================================================================
# DEMONSTRATION & TESTING
# ==============================================================================

def run_demonstration():
    """Comprehensive demonstration of the cognitive system"""
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}HYDRA COGNITIVE SYSTEM - DEMONSTRATION")
    print(f"{Fore.CYAN}{'='*70}\n")
    
    # Initialize Memory System
    mem = EnhancedMemory("demo_memory.json")
    
    # Create Agent Team with distinct personalities
    agents = {
        "STINKMEANER": CognitiveAgent(
            "STINKMEANER", 
            "Chaos Trader", 
            {"aggression": 0.9, "patience": 0.1, "greed": 0.8, "fear": 0.2}, 
            Fore.RED
        ),
        "SAMUEL": CognitiveAgent(
            "SAMUEL",
            "Strategic Leader",
            {"aggression": 0.5, "patience": 0.8, "greed": 0.4, "fear": 0.3},
            Fore.YELLOW
        ),
        "CLAYTON": CognitiveAgent(
            "CLAYTON",
            "Risk Manager",
            {"aggression": 0.2, "patience": 0.9, "greed": 0.3, "fear": 0.7},
            Fore.WHITE
        )
    }
    
    # Seed memory with past experiences
    print(f"{Fore.MAGENTA}📝 Seeding Memory with Past Events...\n")
    
    mem.add_episode(
        "STINKMEANER", 
        "Lost $500 on a fake breakout in DOGE",
        -0.9, 
        ["breakout", "loss", "fakeout", "doge"],
        {"amount": -500, "asset": "DOGE"}
    )
    
    mem.add_episode(
        "SAMUEL",
        "Caught BTC pump at perfect timing, +$1200 profit",
        0.85,
        ["bitcoin", "pump", "win", "timing"],
        {"amount": 1200, "asset": "BTC"}
    )
    
    mem.add_episode(
        "CLAYTON",
        "Avoided ETH crash by waiting for confirmation",
        0.7,
        ["ethereum", "crash", "patience", "wait"],
        {"saved_loss": 800, "asset": "ETH"}
    )
    
    # Test scenarios
    scenarios = [
        "Bitcoin is showing a breakout pattern on the 1H chart with high volume",
        "DOGE pump detected - 40% up in 15 minutes",
        "Ethereum showing bearish divergence, possible crash incoming",
        "Bitcoin support broken, momentum turning negative"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{Fore.CYAN}{'─'*70}")
        print(f"{Fore.CYAN}SCENARIO {i}: {scenario}")
        print(f"{Fore.CYAN}{'─'*70}\n")
        
        # Each agent analyzes the scenario
        for agent_name, agent in agents.items():
            result = agent.think(scenario, mem)
            
            agent.speak(f"Initial Reaction: {result['reaction']}")
            
            if result['memories']:
                print(f"{Fore.YELLOW}  💭 Accessing {len(result['memories'])} relevant memories...")
            
            agent.speak(f"Final Analysis: {result['final_thought']}")
            agent.speak(f"DECISION: {result['decision']} (Confidence: {result['confidence']:.0%})")
            print()
        
        time.sleep(1)
    
    # Show memory statistics
    print(f"\n{Fore.GREEN}{'='*70}")
    print(f"{Fore.GREEN}MEMORY SYSTEM STATISTICS")
    print(f"{Fore.GREEN}{'='*70}")
    stats = mem.get_statistics()
    print(f"{Fore.WHITE}Total Episodes: {stats['total_episodes']}")
    print(f"Average Emotional Impact: {stats['avg_impact']:.2f}")
    print(f"Wins: {stats['total_wins']} | Losses: {stats['total_losses']}")
    print(f"Win Rate: {stats['total_wins']/(stats['total_wins']+stats['total_losses'])*100:.1f}%")
    print(f"{Fore.GREEN}{'='*70}\n")


if __name__ == "__main__":
    run_demonstration()
