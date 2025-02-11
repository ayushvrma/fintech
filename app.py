import streamlit as st
import json
from agent import TradingAgent
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = TradingAgent("config_schema.yaml")
if 'messages' not in st.session_state:
    st.session_state.messages = []

def format_strategy(strategy):
    """Format strategy for display"""
    return f"""
    **Type:** {strategy.get('type', 'N/A')}
    **Timeframe:** {strategy.get('timeframe', 'N/A')}
    
    **Recommendations:**
    {''.join([f"- {rec.get('action', '')} {rec.get('instrument', '')} - {rec.get('rationale', '')}" 
              for rec in strategy.get('recommendations', [])])}
    """

def plot_technical_indicators(market_data):
    """Create technical analysis plot"""
    if not market_data or 'technical' not in market_data:
        return None
    
    tech_data = market_data['technical']
    fig = go.Figure()
    
    # Add price
    fig.add_trace(go.Scatter(
        y=[tech_data.get('current_price', 0)],
        name='Current Price',
        mode='markers'
    ))
    
    # Add SMAs
    fig.add_trace(go.Scatter(
        y=[tech_data.get('sma_20', 0)],
        name='SMA 20',
        mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        y=[tech_data.get('sma_50', 0)],
        name='SMA 50',
        mode='lines'
    ))
    
    fig.update_layout(
        title='Technical Indicators',
        yaxis_title='Price',
        showlegend=True
    )
    
    return fig

# Sidebar - Configuration and History
st.sidebar.title("Trading Assistant")

# Show conversation history
st.sidebar.subheader("Conversation History")
history = st.session_state.agent.get_conversation_history()
for conv in history:
    with st.sidebar.expander(f"💬 {conv['timestamp'][:10]}"):
        st.write(f"Query: {conv['query']}")
        if 'strategies' in conv:
            st.write("Strategies:", format_strategy(conv['strategies']))

# Main chat interface
st.title("Trading Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display technical analysis if available
        if message.get("market_data"):
            fig = plot_technical_indicators(message["market_data"])
            if fig:
                st.plotly_chart(fig)
        
        # Display strategies if available
        if message.get("strategies"):
            st.markdown("### Generated Strategies")
            st.markdown(format_strategy(message["strategies"]))

# Chat input
if prompt := st.chat_input("What would you like to analyze?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = st.session_state.agent.run(prompt)
            
            # Extract relevant information
            market_data = response.get("market_data", {})
            strategies = response.get("strategies", {})
            execution_results = response.get("execution_results", {})
            
            # Display response
            st.markdown("### Market Analysis")
            st.markdown(f"Based on your query, I've analyzed the market conditions.")
            
            # Show technical analysis plot
            fig = plot_technical_indicators(market_data)
            if fig:
                st.plotly_chart(fig)
            
            # Display strategies
            if strategies:
                st.markdown("### Generated Strategies")
                st.markdown(format_strategy(strategies))
            
            # Display execution results if any
            if execution_results and execution_results != {"error": "Strategy violates risk parameters"}:
                st.markdown("### Execution Results")
                st.json(execution_results)
            
            # Add response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I've analyzed the market and generated strategies based on your request.",
                "market_data": market_data,
                "strategies": strategies,
                "execution_results": execution_results
            })

# Additional features
with st.expander("📊 Trading Statistics"):
    # Show basic stats from trading history
    history = st.session_state.agent.get_conversation_history()
    if history:
        trades_df = pd.DataFrame([
            {
                'date': conv['timestamp'],
                'strategy_type': conv.get('strategies', {}).get('type', 'N/A'),
                'status': conv.get('execution_results', {}).get('status', 'N/A')
            }
            for conv in history
        ])
        st.dataframe(trades_df)

with st.expander("⚙️ Configuration"):
    # Show current configuration
    st.json(st.session_state.agent.config) 