from typing import Dict, List, Tuple, Any
from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from tools.market_research import NSEDataTool, MarketNewsTool, TechnicalAnalysisTool, OptionsChainAnalyzer
from tools.strategy import StrategyGenerator, TradeExecutor
from memory_manager import MemoryManager
import yaml
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

class TradingAgent:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = self._initialize_tools()
        self.workflow = self._create_workflow()
        self.memory = MemoryManager()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_tools(self) -> List[Tool]:
        return [
            Tool(
                name="nse_data",
                func=NSEDataTool()._run,
                description="Get NSE market data"
            ),
            Tool(
                name="market_news",
                func=MarketNewsTool()._run,
                description="Get market news and analysis"
            ),
            Tool(
                name="technical_analysis",
                func=TechnicalAnalysisTool()._run,
                description="Perform technical analysis"
            ),
            Tool(
                name="options_analysis",
                func=OptionsChainAnalyzer()._run,
                description="Analyze options chain"
            ),
            Tool(
                name="strategy_generator",
                func=StrategyGenerator()._run,
                description="Generate trading strategies"
            ),
            Tool(
                name="trade_executor",
                func=TradeExecutor()._run,
                description="Execute and log trades"
            )
        ]

    def _create_workflow(self) -> Graph:
        def research(state):
            messages = state["messages"]
            current_message = messages[-1]
            
            # Get relevant past conversations
            past_conversations = self.memory.search_conversations(current_message.content)
            
            # Include past conversations in the prompt
            research_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert Indian market researcher. Analyze the market conditions and gather relevant data. Consider these past relevant conversations: {past_conversations}"),
                ("human", "{input}")
            ])
            
            research_chain = research_prompt | self.llm
            
            # Gather market data
            market_data = {
                "technical": self.tools[2].func("NIFTY"),
                "news": self.tools[1].func("market overview"),
                "options": self.tools[3].func("NIFTY")
            }
            
            return {
                "messages": messages + [AIMessage(content=str(market_data))],
                "market_data": market_data,
                "past_conversations": past_conversations
            }

        def generate_strategy(state):
            messages = state["messages"]
            market_data = state["market_data"]
            past_conversations = state.get("past_conversations", [])
            
            # Get latest checkpoint if exists
            latest_checkpoint = self.memory.get_latest_checkpoint()
            
            strategy_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert Indian F&O trader. Generate trading strategies based on the market data and investment profile. Consider past strategies if relevant: {past_strategies}"),
                ("human", "{market_data}")
            ])
            
            strategy_chain = strategy_prompt | self.llm
            
            strategies = self.tools[4].func(
                market_data["technical"],
                {"fundamental": "data"},
                market_data["options"],
                self.config
            )
            
            # Create checkpoint after strategy generation
            if latest_checkpoint:
                self.memory.create_checkpoint(
                    state.get("conversation_id", "unknown"),
                    {"market_data": market_data, "strategies": strategies},
                    "Strategy generation checkpoint"
                )
            
            return {
                "messages": messages + [AIMessage(content=str(strategies))],
                "strategies": strategies,
                "market_data": market_data
            }

        def execute_trades(state):
            messages = state["messages"]
            strategies = state["strategies"]
            market_data = state["market_data"]
            
            execution_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert trade executor. Execute the generated strategies according to risk parameters."),
                ("human", "{strategies}")
            ])
            
            execution_chain = execution_prompt | self.llm
            
            execution_results = self.tools[5].func(strategies, self.config)
            
            # Save conversation after execution
            conversation_id = self.memory.save_conversation(
                messages[0].content,
                execution_results,
                market_data,
                strategies
            )
            
            return {
                "messages": messages + [AIMessage(content=str(execution_results))],
                "execution_results": execution_results,
                "conversation_id": conversation_id
            }

        workflow = Graph()
        workflow.add_node("research", research)
        workflow.add_node("generate_strategy", generate_strategy)
        workflow.add_node("execute_trades", execute_trades)

        workflow.add_edge("research", "generate_strategy")
        workflow.add_edge("generate_strategy", "execute_trades")
        workflow.add_edge("execute_trades", END)

        return workflow.compile()

    def run(self, query: str) -> Dict:
        try:
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "market_data": {},
                "strategies": {},
                "execution_results": {}
            }
            
            result = self.workflow.invoke(initial_state)
            logger.info("Trading workflow completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error running trading workflow: {e}")
            return {"error": str(e)}

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.memory.get_conversation_history(limit)

    def get_checkpoints(self) -> List[Dict]:
        """Get all checkpoints"""
        return self.memory.get_checkpoints()

if __name__ == "__main__":
    agent = TradingAgent("config_schema.yaml")
    result = agent.run("Analyze market conditions and suggest F&O strategies for NIFTY") 