# 🤖 AI Trading Assistant

An intelligent trading assistant powered by LangChain and GPT-4 that helps you make informed decisions in the Indian stock market, with a focus on F&O trading.

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Features

### 🎯 Core Capabilities
- **Market Analysis**: Real-time analysis of Indian stock market conditions
- **Strategy Generation**: Custom F&O strategies based on your investment profile
- **Risk Management**: Built-in risk parameters and position sizing
- **Trade Execution**: Automated trade execution with proper logging
- **Memory Management**: Contextual awareness of past decisions and market conditions

### 💡 Smart Features
- **Adaptive Learning**: Uses past conversations and decisions to improve recommendations
- **Multi-timeframe Analysis**: Short-term, medium-term, and long-term strategy generation
- **Risk-Aware**: Considers your liquidity needs and risk tolerance
- **Technical Analysis**: Advanced indicators and chart pattern recognition
- **News Integration**: Real-time market news analysis and impact assessment

### 🎨 User Interface
- **Interactive Chat**: Natural conversation interface for market analysis
- **Visual Analytics**: Real-time charts and technical indicators
- **Trading Dashboard**: Track your strategies and execution history
- **Configuration Management**: Easy customization of trading parameters

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- NSE credentials (for live market data)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

4. Configure your investment profile:
```bash
cp config_schema.yaml my_config.yaml
# Edit my_config.yaml with your investment preferences
```

### Running the Application

Start the Streamlit interface:
```bash
streamlit run app.py
```

## 💼 Usage

### 1. Investment Configuration
Configure your investment profile in `config_schema.yaml`:
```yaml
investment_profile:
  total_amount: 1000000  # Total investment amount in INR
  risk_profile: "moderate"  # Options: conservative, moderate, aggressive
  
time_horizons:
  short_term:
    percentage: 30
    liquidity_date: "2024-12-31"
  # ... more configuration options
```

### 2. Market Analysis
Ask the assistant about market conditions:
```
"What's your analysis of NIFTY for the next week considering current volatility?"
```

### 3. Strategy Generation
Request trading strategies:
```
"Generate a hedged F&O strategy for BANKNIFTY with maximum risk of 2%"
```

### 4. Trade Execution
Execute strategies with proper risk management:
```
"Execute the suggested bull call spread on NIFTY with defined parameters"
```

## 🏗️ Architecture

### Components
- **TradingAgent**: Core orchestrator for market analysis and trading
- **MemoryManager**: Handles conversation history and trading checkpoints
- **Market Research Tools**: Real-time data collection and analysis
- **Strategy Generator**: Creates customized trading strategies
- **Trade Executor**: Handles order execution and logging

### Data Flow
1. User Input → Natural Language Processing
2. Market Analysis → Data Collection & Processing
3. Strategy Generation → Risk Assessment
4. Trade Execution → Position Management
5. Memory Storage → Learning & Improvement

## 📊 Features in Detail

### Market Research
- Technical Analysis (RSI, MACD, Moving Averages)
- Options Chain Analysis
- Market News Integration
- Sentiment Analysis

### Strategy Generation
- F&O Strategies (Spreads, Straddles, Iron Condors)
- Position Sizing
- Risk-Reward Calculation
- Market Regime Adaptation

### Risk Management
- Position Size Limits
- Stop-Loss Management
- Portfolio Correlation Analysis
- Volatility Assessment

## 🔄 Future Integrations

### Planned Features
- Groww API Integration for automated trading
- Advanced Portfolio Analytics
- Machine Learning for Pattern Recognition
- Real-time Alerts System
- Performance Attribution Analysis

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## ⚠️ Disclaimer

This software is for educational purposes only. Always do your own research and consider consulting with a financial advisor before making investment decisions. 