# 🧠 HYDRA COGNITIVE TRADING SYSTEM

An advanced multi-agent trading system with episodic memory, chain-of-thought reasoning, and collective decision-making.

## 🌟 Key Features

### 1. **Episodic Memory System**
- **Long-term Memory**: Stores significant trading events with emotional impact scores
- **Associative Retrieval**: Recalls similar past situations when making decisions
- **Memory Decay**: Recent memories weighted more heavily than old ones
- **Trauma/Win Recognition**: Learns from both losses and successes

### 2. **Cognitive Agents with Dual-Process Thinking**
- **System 1 (Fast)**: Immediate gut reactions based on personality
- **System 2 (Slow)**: Deliberate reasoning that integrates memories
- **Personality Traits**: Each agent has unique characteristics (aggression, patience, greed, fear)
- **Chain-of-Thought**: Transparent reasoning process from reaction → memory → decision

### 3. **Council-Based Decision Making**
- **Multiple Perspectives**: Different agents analyze the same market data
- **Consensus Voting**: Democratic decision-making process
- **Confidence Scoring**: Measures certainty of decisions
- **Override Mechanism**: Agents can change their vote based on memories

### 4. **Risk Management**
- **Position Sizing**: Kelly Criterion-inspired calculations
- **Drawdown Limits**: Automatic circuit breakers
- **Volatility Adjustment**: Scales positions based on market conditions
- **Portfolio Protection**: Never risks more than configured maximum

## 📁 File Structure

```
hydra_cognitive_system.py       # Core memory & agent classes
hydra_trading_integration.py    # Market analysis & trading logic
requirements_cognitive.txt       # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_cognitive.txt
```

### Run Demonstration

```bash
# Basic cognitive system demo
python hydra_cognitive_system.py

# Full trading integration demo
python hydra_trading_integration.py
```

## 🧩 Architecture

### Memory System

```python
memory = EnhancedMemory("memory.json")

# Add an episode
memory.add_episode(
    agent="STINKMEANER",
    event="Lost $500 on fake breakout",
    emotional_impact=-0.9,  # -1.0 to 1.0
    tags=["breakout", "loss", "fakeout"],
    metadata={"amount": -500}
)

# Retrieve relevant memories
memories = memory.retrieve_relevant(
    context_tags=["breakout", "bitcoin"],
    limit=3
)
```

### Cognitive Agents

```python
agent = CognitiveAgent(
    name="STINKMEANER",
    role="Chaos Trader",
    personality_traits={
        "aggression": 0.9,  # High risk tolerance
        "patience": 0.1,    # Low patience
        "greed": 0.8,       # High profit seeking
        "fear": 0.2         # Low fear
    },
    color=Fore.RED
)

# Make a decision
result = agent.think(
    context="Bitcoin breakout on 1H chart",
    memory_system=memory
)

print(result['decision'])      # BUY/SELL/HOLD/WAIT
print(result['confidence'])    # 0.0 to 1.0
print(result['final_thought']) # Reasoning chain
```

### Trading Integration

```python
council = HydraCouncil()

# Analyze a trading opportunity
result = await council.analyze_opportunity("BTC/USD")

# Run a full trading session
await council.run_trading_session(
    symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
    cycles=3
)
```

## 🎭 Agent Personalities

### STINKMEANER (The Aggressor)
- **Traits**: High aggression (0.9), Low patience (0.2)
- **Style**: Jumps on opportunities quickly
- **Risk**: Highest risk tolerance
- **Best For**: Catching breakouts, momentum plays

### SAMUEL (The Strategist)
- **Traits**: Balanced aggression (0.5), High patience (0.8)
- **Style**: Methodical, waits for confirmation
- **Risk**: Moderate risk tolerance
- **Best For**: Swing trades, strategic entries

### CLAYTON (The Conservator)
- **Traits**: Low aggression (0.2), Highest patience (0.9)
- **Style**: Ultra-cautious, prioritizes capital preservation
- **Risk**: Lowest risk tolerance
- **Best For**: Risk management, avoiding bad trades

## 📊 How It Works

### Decision-Making Process

1. **Market Analysis**
   - Fetch current price, volume, indicators (RSI, MACD)
   - Detect patterns (overbought, oversold, momentum)

2. **Agent Reasoning** (Each agent independently)
   ```
   Step 1: Initial Reaction (System 1)
   ├─ "Bitcoin breakout detected → BUY"
   │
   Step 2: Memory Retrieval
   ├─ Search for similar past events
   ├─ Found: "Lost $500 on fake breakout (-0.9)"
   │
   Step 3: Synthesis (System 2)
   └─ "WAIT... this is similar to that loss"
       → Override to HOLD
   ```

3. **Council Vote**
   - Aggregate all agent decisions
   - Calculate consensus strength
   - Determine final action

4. **Risk Management**
   - Calculate position size based on confidence
   - Adjust for volatility
   - Check drawdown limits

### Memory Scoring Algorithm

```python
score = (
    (tag_overlap * 2.0) +           # Semantic similarity
    (recency_score) +               # Time decay
    (abs(emotional_impact) * 1.5)   # Intensity
)
```

Higher scores = more relevant memories retrieved

## 🛠️ Customization

### Creating Custom Agents

```python
custom_agent = CognitiveAgent(
    name="CUSTOM",
    role="Your Role",
    personality_traits={
        "aggression": 0.6,
        "patience": 0.5,
        "greed": 0.4,
        "fear": 0.5,
        # Add more traits as needed
    },
    color=Fore.CYAN
)
```

### Adjusting Risk Parameters

```python
risk_manager = RiskManager(
    max_position_size=0.10,  # 10% max per trade
    max_drawdown=0.05        # 5% portfolio stop
)
```

### Memory Configuration

```python
# Control memory size
if len(memory.episodes) > 1000:
    # Keeps 800 most impactful memories
    memory.prune()
```

## 📈 Example Output

```
======================================================================
🎯 ANALYZING: BTC/USD
======================================================================

📊 Market Data:
   Price: $46,835.58
   Volume: $3,587,469
   RSI: 79.7
   MACD: -39.6
   Patterns: overbought, high_volume
   Trend: BEARISH

[01:02:47] [STINKMEANER]: Vote: HOLD (Confidence: 61%)
   Reasoning: WAIT. This reminds me of fake DOGE breakout...

[01:02:47] [SAMUEL]: Vote: HOLD (Confidence: 61%)
   Reasoning: Pattern recognition says wait for confirmation...

[01:02:47] [CLAYTON]: Vote: HOLD (Confidence: 61%)
   Reasoning: Capital preservation mode activated...

──────────────────────────────────────────────────────────────────────
🏛️  COUNCIL DECISION: HOLD
   Consensus Strength: 100%
   Average Confidence: 61%
──────────────────────────────────────────────────────────────────────
```

## 🔬 Advanced Features

### Pattern Learning
The system learns which patterns lead to wins/losses:
- "breakout" + "high_volume" + past loss → Extra caution
- "oversold" + "bullish_momentum" + past win → Increased confidence

### Emotional Impact Scoring
- **+1.0**: Massive win, life-changing trade
- **+0.7**: Strong profitable trade
- **0.0**: Breakeven or insignificant
- **-0.7**: Painful loss
- **-1.0**: Devastating loss, account damage

### Memory Persistence
All memories saved to JSON file:
```json
{
  "episodes": [
    {
      "timestamp": 1707520847.123,
      "agent": "STINKMEANER",
      "event": "Lost $500 on fake breakout",
      "impact": -0.9,
      "tags": ["breakout", "loss", "fakeout"],
      "metadata": {"amount": -500}
    }
  ]
}
```

## ⚠️ Important Notes

### This is a Research/Educational Tool
- **NOT FINANCIAL ADVICE**: This system is for learning and experimentation
- **Paper Trading First**: Always test with simulated money first
- **No Guarantees**: Past performance ≠ future results
- **Risk Disclosure**: Trading cryptocurrencies involves substantial risk

### Security Best Practices
- Never commit API keys to version control
- Use environment variables for credentials
- Rotate keys regularly
- Test with small amounts first

### Limitations
- Simulated market data in demo (replace with real APIs)
- Simple technical indicators (can be enhanced)
- No backtesting framework (add your own)
- No live order execution (intentionally)

## 🔗 Integration Points

To connect to real trading:

1. **Replace Market Analyzer**
   ```python
   # Instead of random data:
   def analyze_market(self, symbol):
       return exchange_api.get_market_data(symbol)
   ```

2. **Add Order Execution**
   ```python
   def execute_trade(self, decision, size):
       if decision == "BUY":
           exchange_api.place_order(
               symbol=symbol,
               side="buy",
               amount=size
           )
   ```

3. **Real-time Data**
   ```python
   # WebSocket connections for live data
   async def stream_market_data():
       async with websocket.connect(url) as ws:
           async for msg in ws:
               await process_tick(msg)
   ```

## 📚 References

This system implements concepts from:
- **Dual-Process Theory**: Kahneman's "Thinking, Fast and Slow"
- **Episodic Memory**: Cognitive psychology research
- **Multi-Agent Systems**: Distributed AI decision-making
- **Kelly Criterion**: Position sizing mathematics
- **Chain-of-Thought**: LLM reasoning techniques applied to trading

## 🤝 Contributing

Ideas for enhancement:
- Add sentiment analysis from news/social media
- Implement proper backtesting framework
- Create web dashboard for monitoring
- Add more sophisticated technical indicators
- Integrate machine learning for pattern recognition
- Build portfolio optimization logic

## 📄 License

This is educational software. Use at your own risk.

---

**Built with**: Python 3.12+, Asyncio, Colorama

**Author**: For research and educational purposes

**Status**: Demo/Prototype - Not production-ready
