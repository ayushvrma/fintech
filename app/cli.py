#!/usr/bin/env python3
"""
CLI utility for testing the financial chatbot.

This tool allows testing the chatbot from the command line without
running the full web server.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

from app.agents.graph import FinancialChatbotGraph
from app.schemas.base import ChatbotResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "False").lower() != "true" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_charts(chart_paths: list):
    """Display information about the generated charts."""
    print("\nGenerated Charts:")
    for i, path in enumerate(chart_paths, 1):
        print(f"  {i}. {os.path.basename(path)}")
        print(f"     Path: {path}")
    
    # Try to detect the environment to suggest viewing options
    if "DISPLAY" in os.environ:
        print("\nTo view the charts:")
        for path in chart_paths:
            print(f"  open {path}")
    else:
        print("\nCharts saved to the paths above. Use an image viewer to open them.")


def format_response(response: ChatbotResponse):
    """Format the chatbot response for display in the terminal."""
    print("\n" + "="*80)
    print(f"RESPONSE ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*80)
    
    print(response.answer)
    
    if response.charts:
        display_charts(response.charts)
    
    if response.error:
        print(f"\nERROR: {response.error}")
    
    print("="*80)


def main():
    """Run the CLI interface."""
    parser = argparse.ArgumentParser(description="Financial Analysis Chatbot CLI")
    parser.add_argument(
        "query",
        nargs="?",
        help="Query for the chatbot (if not provided, interactive mode will be used)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize the chatbot
    print("Initializing financial chatbot...")
    chatbot = FinancialChatbotGraph()
    
    if args.interactive or not args.query:
        print("\nFinancial Analysis Chatbot")
        print("Enter 'exit', 'quit', or Ctrl+C to exit")
        print("="*80)
        
        try:
            while True:
                query = input("\nYou: ")
                if query.lower() in ["exit", "quit"]:
                    break
                
                print("Processing your request, please wait...")
                response = chatbot.process_query(query)
                format_response(response)
                
        except KeyboardInterrupt:
            print("\nExiting...")
        
    else:
        print(f"Processing query: {args.query}")
        response = chatbot.process_query(args.query)
        format_response(response)


if __name__ == "__main__":
    main() 