import os
import logging
from typing import Dict, List, Optional, Any, TypedDict, cast, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain.schema import StrOutputParser
from langraph.graph import END, StateGraph

from app.schemas.graph_state import (
    QueryParsingState, DataRetrievalState, ChartGenerationState,
    AnalysisState, ResponseGenerationState, FinancialChatbotState
)
from app.schemas.base import ChatbotResponse, UserQuery
from app.agents.nodes import (
    query_parsing_node, data_retrieval_node, chart_generation_node,
    technical_analysis_node, response_generation_node
)

logger = logging.getLogger(__name__)


class FinancialChatbotGraph:
    """LangGraph workflow for financial analysis chatbot."""
    
    def __init__(self):
        """Initialize the financial chatbot graph."""
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        builder = StateGraph(FinancialChatbotState)
        
        # Add nodes
        builder.add_node("query_parsing", query_parsing_node)
        builder.add_node("data_retrieval", data_retrieval_node)
        builder.add_node("chart_generation", chart_generation_node)
        builder.add_node("technical_analysis", technical_analysis_node)
        builder.add_node("response_generation", response_generation_node)
        
        # Define edges
        # Start: Parse the query
        builder.set_entry_point("query_parsing")
        
        # Conditional edges based on query type
        # If this is a price check, go to data retrieval then chart generation
        builder.add_conditional_edges(
            "query_parsing",
            lambda state: state.query_parsing.query_type == "price_check",
            {
                True: "data_retrieval",
                False: "data_retrieval"  # Fall through to the other flow
            }
        )
        
        # After data retrieval, generate charts
        builder.add_edge("data_retrieval", "chart_generation")
        
        # Conditional: If this is a technical analysis, perform analysis
        builder.add_conditional_edges(
            "chart_generation",
            lambda state: state.query_parsing.query_type == "analysis" and state.query_parsing.parsed_indicators,
            {
                True: "technical_analysis",
                False: "response_generation"  # Skip analysis for simple price checks
            }
        )
        
        # After analysis, generate response
        builder.add_edge("technical_analysis", "response_generation")
        
        # Response generation is the end of the workflow
        builder.add_edge("response_generation", END)
        
        # Compile the graph
        return builder.compile()
    
    def process_query(self, query: str) -> ChatbotResponse:
        """
        Process a user query through the workflow.
        
        Args:
            query: User's financial question
            
        Returns:
            ChatbotResponse: The chatbot's response including text and charts
        """
        try:
            # Initialize the state
            initial_state = FinancialChatbotState(
                query_parsing=QueryParsingState(raw_query=query)
            )
            
            # Execute the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the response
            if result.response_generation and result.response_generation.generated_response:
                return result.response_generation.generated_response
            
            # If we don't have a valid response, return an error
            return ChatbotResponse(
                answer="I'm sorry, I couldn't process your request. Please try again with a clearer question.",
                error="Workflow did not produce a valid response"
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return ChatbotResponse(
                answer="I encountered an error while processing your request. Please try again later.",
                error=str(e)
            )
    
    def _combine_states(self, state: FinancialChatbotState) -> None:
        """
        Combine the results from different workflow states.
        This ensures data flows properly between different stages of the workflow.
        """
        # Transfer parsed assets to data retrieval
        if state.query_parsing and state.query_parsing.parsed_assets:
            if not state.data_retrieval:
                state.data_retrieval = DataRetrievalState(
                    asset_requests=state.query_parsing.parsed_assets
                )
            else:
                state.data_retrieval.asset_requests = state.query_parsing.parsed_assets
        
        # Transfer retrieved data to chart generation
        if state.data_retrieval and state.data_retrieval.retrieved_data:
            if not state.chart_generation:
                state.chart_generation = ChartGenerationState(
                    price_data=state.data_retrieval.retrieved_data,
                    asset_requests=state.data_retrieval.asset_requests
                )
            else:
                state.chart_generation.price_data = state.data_retrieval.retrieved_data
                state.chart_generation.asset_requests = state.data_retrieval.asset_requests
        
        # Transfer data and indicators to technical analysis
        if state.chart_generation and state.query_parsing.parsed_indicators:
            if not state.analysis:
                state.analysis = AnalysisState(
                    price_data=state.chart_generation.price_data,
                    asset_requests=state.chart_generation.asset_requests,
                    indicators=state.query_parsing.parsed_indicators,
                    parameters=state.query_parsing.parsed_parameters
                )
            else:
                state.analysis.price_data = state.chart_generation.price_data
                state.analysis.asset_requests = state.chart_generation.asset_requests
                state.analysis.indicators = state.query_parsing.parsed_indicators
                state.analysis.parameters = state.query_parsing.parsed_parameters
        
        # Transfer everything to response generation
        if not state.response_generation:
            state.response_generation = ResponseGenerationState(
                raw_query=state.query_parsing.raw_query,
                charts=state.chart_generation.generated_charts if state.chart_generation else None,
                analysis_results=state.analysis.analysis_results if state.analysis else None
            )
        else:
            state.response_generation.raw_query = state.query_parsing.raw_query
            if state.chart_generation:
                state.response_generation.charts = state.chart_generation.generated_charts
            if state.analysis:
                state.response_generation.analysis_results = state.analysis.analysis_results 