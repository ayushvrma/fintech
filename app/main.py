import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Body
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import json

from app.schemas.base import UserQuery, ChatbotResponse, AssetType, TimeFrame, TechnicalIndicator
from app.agents.graph import FinancialChatbotGraph

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "False").lower() != "true" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Financial Analysis Chatbot",
    description="A chatbot that provides real-time financial analysis for stocks and cryptocurrencies",
    version="1.0.0"
)

# Initialize the chatbot graph
chatbot_graph = FinancialChatbotGraph()

# Charts directory for static files
CHART_DIR = os.path.join(os.path.dirname(__file__), 'data', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)
app.mount("/charts", StaticFiles(directory=CHART_DIR), name="charts")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Return the welcome page."""
    return """
    <html>
        <head>
            <title>Financial Analysis Chatbot</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                }
                .container {
                    width: 80%;
                    max-width: 800px;
                    background-color: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    margin-bottom: 1rem;
                }
                p {
                    color: #34495e;
                    line-height: 1.6;
                    margin-bottom: 1.5rem;
                }
                .endpoints {
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 4px;
                    margin-top: 2rem;
                }
                .endpoint {
                    margin-bottom: 0.5rem;
                    font-family: monospace;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Financial Analysis Chatbot API</h1>
                <p>Welcome to the Financial Analysis Chatbot API. This service provides real-time financial data and technical analysis for stocks and cryptocurrencies.</p>
                
                <p>You can interact with this API by sending requests to the following endpoints:</p>
                
                <div class="endpoints">
                    <div class="endpoint"><strong>POST /chat</strong> - Send a question to the chatbot</div>
                    <div class="endpoint"><strong>GET /charts/{filename}</strong> - Retrieve a generated chart</div>
                    <div class="endpoint"><strong>GET /docs</strong> - OpenAPI documentation</div>
                </div>
                
                <p>For more detailed information, please check the <a href="/docs">API documentation</a>.</p>
            </div>
        </body>
    </html>
    """

@app.post("/chat", response_model=ChatbotResponse)
async def chat(query: UserQuery):
    """
    Process a user query and return a response.
    
    Args:
        query (UserQuery): The user's question
        
    Returns:
        ChatbotResponse: The chatbot's response with analysis and charts
    """
    try:
        logger.info(f"Received query: {query.query}")
        
        # Process the query
        response = chatbot_graph.process_query(query.query)
        
        # Convert chart paths to URLs
        if response.charts:
            response.charts = [
                f"/charts/{os.path.basename(chart_path)}" 
                for chart_path in response.charts
            ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/{filename}")
async def get_chart(filename: str):
    """
    Retrieve a chart by filename.
    
    Args:
        filename (str): The filename of the chart
        
    Returns:
        FileResponse: The chart image
    """
    file_path = os.path.join(CHART_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Chart not found")
    
    return FileResponse(file_path)

@app.get("/asset_types", response_model=List[str])
async def get_asset_types():
    """Get all available asset types."""
    return [asset_type.value for asset_type in AssetType]

@app.get("/timeframes", response_model=List[str])
async def get_timeframes():
    """Get all available timeframes."""
    return [timeframe.value for timeframe in TimeFrame]

@app.get("/indicators", response_model=List[str])
async def get_indicators():
    """Get all available technical indicators."""
    return [indicator.value for indicator in TechnicalIndicator]

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Run app with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    ) 