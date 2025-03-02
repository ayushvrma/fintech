# Financial Analysis Chatbot

A real-time financial analysis chatbot that retrieves stock and cryptocurrency data from multiple sources and performs technical analysis using LangGraph.

## Features

- Real-time data retrieval from financial APIs (Alpha Vantage, Yahoo Finance, Binance, CCXT)
- Chart visualization from multiple platforms
- Automated technical analysis using LangGraph agents
- Interactive chat interface for queries

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   ```
4. Run the application:
   ```bash
   python app/main.py
   ```

## Usage

Ask the chatbot questions about stocks or cryptocurrencies, such as:
- "Show me AAPL stock price for the last week"
- "Perform RSI analysis on Bitcoin for the last month"
- "Compare TSLA and MSFT performance over the last quarter"
- "Show me support and resistance levels for Ethereum"

## Architecture

This project uses LangGraph for agent coordination and workflow management. The system is built with:
- Data retrieval tools for accessing financial APIs
- Visualization components for chart generation
- Technical analysis agents for processing financial data
- Pydantic models for structured data handling 